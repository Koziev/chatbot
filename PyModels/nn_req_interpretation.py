# -*- coding: utf-8 -*-
"""
Тренировка модели для определения необходимости интерпретации (раскрытия анафоры, гэппинга и т.д.)
в реплике пользователя.

Для проекта чатбота https://github.com/Koziev/chatbot
"""

from __future__ import division
from __future__ import print_function

import io
import json
import os
import argparse
import random
import logging

import numpy as np
import pandas as pd
import tqdm

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
import keras.regularizers

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sklearn.metrics

from utils.tokenizer import Tokenizer
from utils.padding_utils import lpad_wordseq, rpad_wordseq
from utils.padding_utils import PAD_WORD
from trainers.word_embeddings import WordEmbeddings
import utils.console_helpers
import utils.logging_helpers


padding = 'left'

random.seed(123456789)
np.random.seed(123456789)


class Sample:
    def __init__(self, phrase, y):
        assert(len(phrase) > 0)
        assert(y in [0, 1])
        self.phrase = phrase
        self.y = y


def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


def load_dataset(params):
    tokenizer = Tokenizer()
    tokenizer.load()

    # Датасет должен быть заранее сформирован скриптом ./preparation/prepare_req_interpretation_classif.py
    df = pd.read_csv(os.path.join(data_folder, 'req_interpretation_dataset.csv'), sep='\t', encoding='utf-8')
    samples = [Sample(row['text'], int(row['label'])) for i, row in df.iterrows()]

    # Токенизация сэмплов
    for sample in samples:
        sample.words = tokenizer.tokenize(sample.phrase)

    nb_0 = sum(sample.y == 0 for sample in samples)
    nb_1 = sum(sample.y == 1 for sample in samples)
    logging.info('nb_0={} nb_1={}'.format(nb_0, nb_1))

    max_wordseq_len = max(len(sample.words) for sample in samples)
    logging.info('max_wordseq_len={}'.format(max_wordseq_len))

    if params['padding'] == 'left':
        for sample in samples:
            sample.words = lpad_wordseq(sample.words, max_wordseq_len)
    else:
        for sample in samples:
            sample.words = rpad_wordseq(sample.words, max_wordseq_len)

    computed_params = {'max_wordseq_len': max_wordseq_len,
                       'nb_0': nb_0,
                       'nb_1': nb_1}

    return samples, computed_params


def create_model(params, computed_params):
    logging.info('Constructing the NN model arch={}...'.format(params['net_arch']))
    max_wordseq_len = computed_params['max_wordseq_len']

    input_words = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_words')

    # суммарный размер выходных тензоров в conv1, то есть это сумма размеров векторов
    # для всех слоев в списке conv1, если их смерджить.
    layers = []
    if params['net_arch'] == 'rnn':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов предложения.
        rnn_size = params['rnn_size']
        words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                 input_shape=(max_wordseq_len, word_dims),
                                                 return_sequences=False))

        encoder_rnn = words_rnn(input_words)
        layers.append(encoder_rnn)
    else:
        raise NotImplementedError()

    if len(layers) == 1:
        classif = layers[0]
    else:
        classif = keras.layers.concatenate(inputs=layers)

    classif = Dense(units=params['units1'], activation=params['activation1'])(classif)
    classif = Dense(units=2, activation='softmax', name='output')(classif)
    model = Model(inputs=input_words, outputs=classif)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    model.summary()
    return model



def vectorize_words(words, X_batch, irow, word2vec):
    for iword, word in enumerate(words):
        if word != PAD_WORD:
            X_batch[irow, iword, :] = word2vec[word]


def generate_rows(samples, batch_size, computed_params, mode):
    if mode == 1:
        # При обучении сетки каждую эпоху тасуем сэмплы.
        random.shuffle(samples)

    batch_index = 0
    batch_count = 0

    w2v = computed_params['embeddings']
    max_wordseq_len = computed_params['max_wordseq_len']

    X_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    while True:
        for irow, sample in enumerate(samples):
            vectorize_words(sample.words, X_batch, batch_index, w2v)
            if mode == 1:
                y_batch[batch_index, sample.y] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                xx = {'input_words': X_batch}
                if mode == 1:
                    yield (xx, {'output': y_batch})
                else:
                    yield xx

                # очищаем матрицы порции для новой порции
                X_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


def train_model(model, train_samples, val_samples, params, computed_params):
    logging.info('Train model with params={}'.format(get_params_str(params)))
    batch_size = params['batch_size']
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=1,
                                   mode='auto')

    nb_validation_steps = len(val_samples) // batch_size

    hist = model.fit_generator(generator=generate_rows(train_samples, batch_size, computed_params, 1),
                               steps_per_epoch=len(train_samples) // batch_size,
                               epochs=1000,
                               verbose=2,
                               callbacks=[model_checkpoint, early_stopping],
                               validation_data=generate_rows(val_samples, batch_size, computed_params, 1),
                               validation_steps=nb_validation_steps,
                               )
    max_acc = max(hist.history['val_acc'])
    logging.info('max val_acc={}'.format(max_acc))

    # загрузим чекпоинт с оптимальными весами
    model.load_weights(weights_path)


def score_model(model, val_samples, params, computed_params):
    logging.info('Estimating final f1 score...')
    # получим оценку F1 на валидационных данных
    batch_size = params['batch_size']
    nb_validation_steps = len(val_samples) // params['batch_size']
    y_true2 = []
    y_pred2 = []
    y_pred3 = []
    for istep, xy in enumerate(generate_rows(val_samples, batch_size, computed_params, 1)):
        x = xy[0]
        y = xy[1]['output']
        y_pred = model.predict(x=x, verbose=0)
        for k in range(len(y_pred)):
            y_true2.append(y[k][1])
            y_pred2.append(y_pred[k][1] > y_pred[k][0])
            y_pred3.append(y_pred[k][1])

        if istep >= nb_validation_steps:
            break

    f1 = sklearn.metrics.f1_score(y_true=y_true2, y_pred=y_pred2)
    logloss = -sklearn.metrics.log_loss(y_true=y_true2, y_pred=y_pred3)
    logging.info('val logloss={} f1={}'.format(logloss, f1))
    return f1


def report_model(model, eval_samples, params, computed_params, output_path):
    # Для отладки - прогоним весь набор данных через модель и сохраним
    # результаты классификации в файл для визуальной проверки.
    with io.open(output_path, 'w', encoding='utf-8') as wrt:
        for sample in tqdm.tqdm(samples, total=len(samples), desc='Predicting'):
            for istep, xy in enumerate(generate_rows([sample], 1, computed_params, 1)):
                x = xy[0]
                y = xy[1]['output']
                y_pred = model.predict(x=x, verbose=0)
                y_true2 = y[0][1]
                y_pred2 = y_pred[0][1] > y_pred[0][0]
                wrt.write(u'{:<80s} y_true={} y_pred={}\n'.format(sample.phrase, y_true2, y_pred2))
                break


# -------------------------------------------------------------------

# Разбор параметров тренировки в командной строке
parser = argparse.ArgumentParser(description='Neural model for interpretation requirement classifier')
parser.add_argument('--run_mode', type=str, default='train', choices='train gridsearch'.split(), help='what to do: train | gridsearch')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing data files')

args = parser.parse_args()
data_folder = args.data_dir
tmp_folder = args.tmp

wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_req_interpretation.log'))

run_mode = args.run_mode

config_path = os.path.join(tmp_folder, 'nn_req_interpretation.config')
arch_filepath = os.path.join(tmp_folder, 'nn_req_interpretation.arch')
weights_path = os.path.join(tmp_folder, 'nn_req_interpretation.weights')

if run_mode == 'gridsearch':
    logging.info('Start gridsearch')

    params = dict()
    best_params = None
    best_score = -np.inf
    crossval_count = 0

    best_score_wrt = open(os.path.join(tmp_folder, 'nn_req_interpretation.best_score.txt'), 'w')

    for padding in ['left']:
        params['padding'] = padding

        samples, computed_params = load_dataset(params)
        embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
        word_dims = embeddings.vector_size
        computed_params['embeddings'] = embeddings
        computed_params['word_dims'] = word_dims

        for net_arch in ['rnn']:
            params['net_arch'] = net_arch

            for rnn_size in [256]:
                params['rnn_size'] = rnn_size

                for units1 in [16]:
                    params['units1'] = units1

                    for activation1 in ['sigmoid', 'relu']:
                        params['activation1'] = activation1

                        for optimizer in ['nadam']:
                            params['optimizer'] = optimizer

                            for batch_size in [150]:
                                params['batch_size'] = batch_size

                                crossval_count += 1
                                logging.info('Start crossvalidation #{} for params={}'.format(crossval_count,
                                                                                              get_params_str(params)))

                                kf = KFold(n_splits=3)
                                scores = []
                                for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
                                    logging.info('KFold[{}]'.format(ifold))

                                    train_samples = [samples[i] for i in train_index]
                                    val12_samples = [samples[i] for i in val_index]
                                    val_samples, eval_samples = train_test_split(val12_samples,
                                                                                 test_size=0.5,
                                                                                 random_state=123456)
                                    model = create_model(params, computed_params)

                                    logging.info('train_samples.count={}'.format(len(train_samples)))
                                    logging.info('val_samples.count={}'.format(len(val_samples)))
                                    logging.info('eval_samples.count={}'.format(len(eval_samples)))
                                    train_model(model, train_samples, val_samples, params, computed_params)

                                    # получим оценку F1 на валидационных данных
                                    logging.info('Estimating final f1 score...')
                                    score = score_model(model, eval_samples, params, computed_params)
                                    logging.info('eval score={}'.format(score))

                                    logging.info('KFold[{}] score={}'.format(ifold, score))
                                    scores.append(score)

                                score = np.mean(scores)
                                score_std = np.std(scores)
                                logging.info('Crossvalidation #{} score={} std={}'.format(crossval_count, score, score_std))
                                if score > best_score:
                                    best_params = params.copy()
                                    best_score = score
                                    logging.info(
                                        '!!! NEW BEST score={} params={}'.format(best_score, get_params_str(best_params)))
                                    best_score_wrt.write('best score={}\nparams={}\n\n'.format(best_score, get_params_str(best_params)))
                                    best_score_wrt.flush()

    logging.info('Grid search complete, best_score={} best_params={}'.format(best_score, get_params_str(best_params)))
    best_score_wrt.close()


if run_mode == 'train':
    logging.info('Start with run_mode==train')

    params = dict()

    params['padding'] = 'left'

    samples, computed_params = load_dataset(params)

    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
    word_dims = embeddings.vector_size
    computed_params['embeddings'] = embeddings
    computed_params['word_dims'] = word_dims

    params['net_arch'] = 'rnn'
    params['rnn_size'] = word_dims*2
    params['units1'] = 32
    params['activation1'] = 'relu'
    params['optimizer'] = 'nadam'
    params['batch_size'] = 150

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'model': 'nn',
        'max_wordseq_len': computed_params['max_wordseq_len'],
        'w2v_path': word2vector_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'padding': params['padding'],
        'arch_filepath': arch_filepath,
        'weights_path': weights_path,
        'word_dims': word_dims,
        'net_arch': params['net_arch'],
    }

    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=4)

    model = create_model(params, computed_params)

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    # Тренировка модели, затем валидация
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=123456)
    val_samples, eval_samples = train_test_split(val_samples, test_size=0.5, random_state=123456)
    logging.info('train_samples.count={}'.format(len(train_samples)))
    logging.info('val_samples.count={}'.format(len(val_samples)))
    logging.info('eval_samples.count={}'.format(len(eval_samples)))
    train_model(model, train_samples, val_samples, params, computed_params)

    # получим оценку F1 на валидационных данных
    logging.info('Estimating final f1 score...')
    score = score_model(model, eval_samples, params, computed_params)
    logging.info('eval score={}'.format(score))

    # Для отладки - прогоним весь набор данных через модель и сохраним
    # результаты классификации в файл для визуальной проверки.
    report_model(model, eval_samples, params, computed_params,
                 os.path.join(tmp_folder, 'nn_req_interpretation.validation.txt'))


if run_mode == 'query':
    # загружаем ранее натренированную сетку и остальные параметры модели.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_wordseq_len = int(model_config['max_wordseq_len'])
        word2vector_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        arch_filepath = model_config['arch_filepath']
        weights_path = model_config['weights_path']
        word_dims = model_config['word_dims']
        net_arch = model_config['net_arch']

    print('Restoring model architecture from {}'.format(arch_filepath))
    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    print('Loading model weights from {}'.format(weights_path))
    model.load_weights(weights_path)

    # TODO ...

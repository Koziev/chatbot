# -*- coding: utf-8 -*-
'''
Тренировка модели классификации yes/no для сэмплов с несколькими (от 0 до n)
предпосылками и вопросом.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
Используется нейросетка (Keras).
Датасет "pqa_yes_no.dat" должен быть сгенерирован и находится в папке ../data (см. prepare_qa_dataset.py)
Также нужна модель встраивания слов (word2vector) и модель посимвольного встраивания (wordchar2vector).
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import sys
import gensim
import codecs
import keras.callbacks
import numpy as np
import tqdm
import argparse
import logging

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics

from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers


PAD_WORD = u''
padding = 'left'



def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer


def pad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def load_embeddings(tmp_folder, word2vector_path):
    wordchar2vector_path = os.path.join(tmp_folder, 'wordchar2vector.dat')
    logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    logging.info('wc2v_dims={0}'.format(wc2v_dims))

    logging.info(u'Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    logging.info('w2v_dims={0}'.format(w2v_dims))

    word_dims = w2v_dims + wc2v_dims
    computed_params['word_dims'] = word_dims

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros(word_dims)
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    del w2v
    #del wc2v
    gc.collect()

    return word2vec, word_dims, wordchar2vector_path


def load_samples(input_path, tokenizer):
    # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
    logging.info(u'Loading samples from {}'.format(input_path))
    samples = []
    nb_yes = 0 # кол-во ответов "да"
    nb_no = 0 # кол-во ответов "нет"
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    max_inputseq_len = 0  # макс. длина предпосылок и вопроса в словах

    with codecs.open(input_path, 'r', 'utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if len(line) == 0:
                if len(lines) > 0:
                    premises = lines[:-2]
                    question = lines[-2]
                    answer = lines[-1]
                    sample = Sample(premises, question, answer)
                    samples.append(sample)

                    max_nb_premises = max(max_nb_premises, len(premises))

                    if answer == u'да':
                        nb_yes += 1
                    elif answer == u'нет':
                        nb_no += 1

                    for phrase in lines:
                        words = tokenizer.tokenize(phrase)
                        max_inputseq_len = max(max_inputseq_len, len(words))

                    lines = []

            else:
                lines.append(line)

    logging.info('samples.count={}'.format(len(samples)))
    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_nb_premises={}'.format(max_nb_premises))
    logging.info('nb_yes={}'.format(nb_yes))
    logging.info('nb_no={}'.format(nb_no))

    computed_params = {'max_nb_premises': max_nb_premises,
                       'max_inputseq_len': max_inputseq_len,
                       }

    return samples, computed_params


def vectorize_words(words, M, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def create_model(params, computed_params):
    net_arch = params['net_arch']
    logging.info('Constructing neural net: {}...'.format(net_arch))

    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    max_nb_premises = computed_params['max_nb_premises']

    inputs = []
    input_question = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='question')
    inputs.append(input_question)

    for ipremise in range(max_nb_premises):
        input_premise = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='premise{}'.format(ipremise))
        inputs.append(input_premise)

    layers = []
    encoder_size = 0

    if net_arch == 'lstm':
        rnn_size = params['rnn_size']

        # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения. Этот слой общий для всех входных предложений.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        for input in inputs:
            encoder_rnn = shared_words_rnn(input)
            layers.append(encoder_rnn)
            encoder_size += rnn_size*2
    elif net_arch == 'lstm(cnn)':
        rnn_size = params['rnn_size']
        nb_filters = params['nb_filters']
        max_kernel_size = params['max_kernel_size']

        for kernel_size in range(1, max_kernel_size+1):
            # сначала идут сверточные слои, образующие детекторы словосочетаний
            # и синтаксических конструкций
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            lstm = recurrent.LSTM(rnn_size, return_sequences=False)

            for input in inputs:
                conv_layer1 = conv(input)

                if params['pooling'] == 'max':
                    pooling = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')
                elif params['pooling'] == 'average':
                    pooling = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')
                else:
                    raise NotImplementedError()

                conv_layer1 = pooling(conv_layer1)

                conv_layer1 = lstm(conv_layer1)
                layers.append(conv_layer1)
                encoder_size += rnn_size
    else:
        raise NotImplementedError()

    encoder_merged = keras.layers.concatenate(inputs=list(layers))
    encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_merged)

    decoder = encoder_final

    if params['units1'] > 0:
        decoder = Dense(encoder_size//2, activation='relu')(decoder)

        if params['units2'] > 0:
            decoder = Dense(encoder_size//4, activation='relu')(decoder)

    #decoder = BatchNormalization()(decoder)

    output_dims = 2
    decoder = Dense(output_dims, activation='softmax', name='output')(decoder)

    model = Model(inputs=inputs, outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    #model.summary()

    return model


def generate_rows(params, computed_params, samples, batch_size, mode):
    batch_index = 0
    batch_count = 0
    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    nb_premises = computed_params['max_nb_premises']

    Xn_batch = []
    for _ in range(nb_premises + 1):
        x = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_batch.append(x)

    inputs = {}
    for ipremise in range(nb_premises):
        inputs['premise{}'.format(ipremise)] = Xn_batch[ipremise]
    inputs['question'] = Xn_batch[nb_premises]

    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    weights = np.zeros((batch_size), dtype=np.float32)
    weights.fill(1.0)

    while True:
        for irow, sample in enumerate(samples):
            for ipremise, premise in enumerate(sample.premises):
                words = tokenizer.tokenize(premise)
                words = pad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[ipremise], batch_index, word2vec)

            words = tokenizer.tokenize(sample.question)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[nb_premises], batch_index, word2vec)

            if sample.answer == u'нет':
                y_batch[batch_index, 0] = True
                weights[batch_index] = 1.0  #float(nb_no+nb_yes) / nb_yes
            else:
                y_batch[batch_index, 1] = True
                weights[batch_index] = 1.0  #float(nb_no+nb_yes) / nb_no

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield (inputs, {'output': y_batch}, weights)
                else:
                    yield inputs

                # очищаем матрицы порции для новой порции
                for x in Xn_batch:
                    x.fill(0)
                y_batch.fill(0)
                batch_index = 0


def train_model(model, params, computed_params, train_samples, val_samples):
    nb_train_patterns = len(train_samples)
    nb_valid_patterns = len(val_samples)
    logging.info('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_acc'
    model_checkpoint = ModelCheckpoint(weights_path,
                                       monitor=monitor_metric,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=5, verbose=1, mode='auto')
    callbacks = [model_checkpoint, early_stopping]

    batch_size = params['batch_size']
    hist = model.fit_generator(generator=generate_rows(params, computed_params, train_samples, batch_size, 1),
                               steps_per_epoch=nb_train_patterns // batch_size,
                               epochs=100,
                               verbose=2,
                               callbacks=callbacks,
                               validation_data=generate_rows(params, computed_params, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns // batch_size)
    logging.info('max val_acc={}'.format(max(hist.history['val_acc'])))
    model.load_weights(weights_path)


def score_model(model, params, computed_params, val_samples, save_report):
    # прогоним валидационные паттерны через модель, чтобы получить f1 score.
    max_nb_premises = computed_params['max_nb_premises']
    nb_valid_patterns = len(val_samples)
    for v in generate_rows(params, computed_params, val_samples, nb_valid_patterns, 1):
        x = v[0]
        y_val = v[1]['output']
        break

    n0 = sum(y_val[:, 1] == 0)
    n1 = sum(y_val[:, 1] == 1)

    logging.info('targets: n0={} n1={}'.format(n0, n1))

    y_pred0 = model.predict(x)[:, 1]
    y_pred = (y_pred0 >= 0.5).astype(np.int)
    f1 = sklearn.metrics.f1_score(y_true=y_val[:, 1], y_pred=y_pred)
    logging.info('val f1={}'.format(f1))

    #score = -sklearn.metrics.log_loss(y_true=y_val[:, 1], y_score=y_pred0)
    score = sklearn.metrics.roc_auc_score(y_true=y_val[:, 1], y_score=y_pred0)
    logging.info('score={}'.format(score))

    if save_report:
        # Сохраним в текстовом файле для визуальной проверки результаты валидации по всем сэмплам
        for v in generate_rows(params, computed_params, samples, len(samples), 1):
            x = v[0]
            break

        y_pred = model.predict(x)[:, 1]
        y_pred = (y_pred >= 0.5).astype(np.int)

        with codecs.open(os.path.join(tmp_folder, 'nn_yes_no.validation.txt'), 'w', 'utf-8') as wrt:
            for isample, sample in enumerate(samples):
                if isample > 0:
                    wrt.write('\n\n')

                for premise in sample.premises:
                    wrt.write(u'P: {}\n'.format(premise))
                wrt.write(u'Q: {}\n'.format(sample.question))
                wrt.write(u'A: {}\n'.format(sample.answer))

                pred = u'да' if y_pred[isample] else u'нет'
                wrt.write(u'model: {}\n'.format(pred))

                #if pred == u'нет':
                #    for ipremise in range(max_nb_premises):
                #        wrt.write('\nX[{}]={}\n'.format(ipremise, x['premise{}'.format(ipremise)][isample]))
                #    wrt.write('\n')

    return f1, score

# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Neural model for yes/no answer classification')
parser.add_argument('--run_mode', type=str, default='train', choices='train gridsearch query'.split(), help='what to do: train | query | gridsearch')
parser.add_argument('--batch_size', type=int, default=250, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/pqa_yes_no.dat', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

args = parser.parse_args()

data_folder = args.data_dir
input_path = args.input
tmp_folder = args.tmp

wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)
batch_size = args.batch_size
run_mode = args.run_mode

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_yes_no.log'))

# В этих файлах будем сохранять натренированную сетку
config_path = os.path.join(tmp_folder, 'nn_yes_no.config')
arch_filepath = os.path.join(tmp_folder, 'nn_yes_no.arch')
weights_path = os.path.join(tmp_folder, 'nn_yes_no.weights')


if run_mode == 'gridsearch':
    logging.info('Start gridsearch')

    tokenizer = Tokenizer()
    tokenizer.load()
    samples, computed_params = load_samples(input_path, tokenizer)

    word2vec, word_dims, wordchar2vector_path = load_embeddings(tmp_folder, word2vector_path)

    best_params = None
    best_score = -np.inf

    params = dict()
    crossval_count = 0
    for net_arch in ['lstm']:
        params['net_arch'] = net_arch

        for rnn_size in [16, 24, 32]:
            params['rnn_size'] = rnn_size

            for units1 in [10, 32]:
                params['units1'] = units1

                for units2 in [0, 5, 10]:
                    params['units2'] = units2

                    for batch_size in [250]:
                        params['batch_size'] = batch_size

                        for optimizer in ['nadam']:
                            params['optimizer'] = optimizer

                            crossval_count += 1
                            logging.info('Crossvalidation #{} for {}'.format(crossval_count, get_params_str(params)))

                            kf = KFold(n_splits=3)
                            scores = []
                            for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
                                logging.info('KFold[{}]'.format(ifold))
                                train_samples = [samples[i] for i in train_index]
                                val12_samples = [samples[i] for i in val_index]

                                SEED = 123456
                                TEST_SHARE = 0.2
                                val_samples, finval_samples = train_test_split(val12_samples, test_size=0.5,
                                                                               random_state=SEED)

                                model = create_model(params, computed_params)
                                train_model(model, params, computed_params, train_samples, val_samples)

                                f1_score, score = score_model(model, params, computed_params, finval_samples, save_report=False)
                                scores.append(score)

                            score = np.mean(scores)
                            score_std = np.std(scores)
                            logging.info('Crossvalidation #{} score={} std={}'.format(crossval_count, score, score_std))
                            if score > best_score:
                                best_params = params.copy()
                                best_score = score
                                logging.info('!!! NEW BEST score={} params={}'.format(best_score, get_params_str(best_params)))

    logging.info('Grid search complete, best_score={} best_params={}'.format(best_score, get_params_str(best_params)))



if run_mode == 'train':
    logging.info('Start run_mode==train')

    tokenizer = Tokenizer()
    tokenizer.load()
    samples, computed_params = load_samples(input_path, tokenizer)

    word2vec, word_dims, wordchar2vector_path = load_embeddings(tmp_folder, word2vector_path)

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'engine': 'nn',
                    'max_inputseq_len': computed_params['max_inputseq_len'],
                    'max_nb_premises': computed_params['max_nb_premises'],
                    'w2v_path': word2vector_path,
                    'wordchar2vector_path': wordchar2vector_path,
                    'PAD_WORD': PAD_WORD,
                    'padding': padding,
                    'model_folder': tmp_folder,
                    'word_dims': word_dims,
                    'arch_filepath': arch_filepath,
                    'weights_filepath': weights_path
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    params = dict()
    params['net_arch'] = 'lstm'
    params['rnn_size'] = 16
    params['units1'] = 10
    params['units2'] = 0
    params['batch_size'] = 250
    params['optimizer'] = 'nadam'

    model = create_model(params, computed_params)

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    SEED = 123456
    TEST_SHARE = 0.2
    train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)

    train_model(model, params, computed_params, train_samples, val_samples)

    f1_score, logloss = score_model(model, params, computed_params, val_samples, save_report=True)



if run_mode == 'query':
    # Ручное консольное тестирование модели, натренированной ранее с помощью --run_mode train

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_inputseq_len = model_config['max_inputseq_len']
        w2v_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        max_nb_premises = model_config['max_nb_premises']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    print('wc2v_dims={0}'.format(wc2v_dims))

    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros( word_dims )
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    del w2v
    gc.collect()

    tokenizer = Tokenizer()
    tokenizer.load()

    Xn_probe = []
    for _ in range(max_nb_premises+1):
        x = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_probe.append(x)

    inputs = dict()
    for ipremise in range(max_nb_premises):
        inputs['premise{}'.format(ipremise)] = Xn_probe[ipremise]
    inputs['question'] = Xn_probe[max_nb_premises]

    while True:
        print('\nEnter 0 to {} premises and one question:'.format(max_nb_premises))
        premises = []
        question = None
        for ipremise in range(max_nb_premises):
            premise = raw_input('premise #{} :> '.format(ipremise)).decode(sys.stdout.encoding).strip().lower()
            if len(premise) == 0:
                break
            if premise[-1] == u'?':
                question = premise
                break

            premises.append(premise)

        if question is None:
            question = raw_input('question:> ').decode(sys.stdout.encoding).strip().lower()
        if len(question) == 0:
            break

        for i in range(max_nb_premises+1):
            Xn_probe[i].fill(0)

        for ipremise, premise in enumerate(premises):
            words = tokenizer.tokenize(premise)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_probe[ipremise], 0, word2vec)

        words = tokenizer.tokenize(question)
        words = pad_wordseq(words, max_inputseq_len)
        vectorize_words(words, Xn_probe[max_nb_premises], 0, word2vec)

        y_probe = model.predict(x=inputs)

        print('p(no) ={}'.format(y_probe[0][0]))
        print('p(yes)={}'.format(y_probe[0][1]))

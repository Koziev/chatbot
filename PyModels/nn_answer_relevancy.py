# -*- coding: utf-8 -*-
"""
Тренировка модели для проверки релевантности ответа для заданной
предпосылки и вопроса. Нужна для ранжирования сгенерированных
вариантов ответа.

Для вопросно-ответной системы https://github.com/Koziev/chatbot.

Используется нейросетка (Keras).

Датасет "answer_relevancy_dataset.dat" должен быть сгенерирован и находится в папке ../data (см. prepare_answer_relevancy_dataset.py).
Также нужна модель встраивания слов (word2vector) и модель посимвольного встраивания (wordchar2vector).
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import gensim
import io
import math
import numpy as np
import argparse
import logging
import random

import keras.callbacks
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


bad_tokens = set(u'? . !'.split())


def clean_phrase(s, tokenizer):
    return u' '.join(filter(lambda w: w not in bad_tokens, tokenizer.tokenize(s))).lower()


class Sample:
    def __init__(self, premises, question, answer, label):
        assert(label in (0, 1))
        assert(len(answer) > 0)
        self.premises = premises[:]
        self.question = question
        self.answer = answer
        self.label = label


def pad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def load_embeddings(tmp_folder, word2vector_path, computed_params):
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


def load_samples(input_path):
    # Загружаем датасеты, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
    samples = []
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    max_inputseq_len = 0  # макс. длина предпосылок и вопроса в словах
    all_answers = set()

    logging.info(u'Loading samples from {}'.format(input_path))

    with io.open(input_path, 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if len(line) == 0:
                if len(lines) > 0:
                    if len(lines) == 4:  # пока берем только сэмплы с 1 предпосылкой
                        premises = lines[:-3]
                        question = lines[-3]
                        answer = lines[-2]
                        label = int(lines[-1])

                        if answer not in (u'да', u'нет'):
                            max_nb_premises = max(max_nb_premises, len(premises))
                            for phrase in lines:
                                words = tokenizer.tokenize(phrase)
                                max_inputseq_len = max(max_inputseq_len, len(words))

                            sample = Sample(premises, question, answer, label)
                            samples.append(sample)
                            all_answers.add(answer)

                    lines = []
            else:
                lines.append(line)

    logging.info('samples.count={}'.format(len(samples)))
    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_nb_premises={}'.format(max_nb_premises))

    nb_0 = sum((sample.label==0) for sample in samples)
    nb_1 = sum((sample.label==1) for sample in samples)

    logging.info('nb_0={}'.format(nb_0))
    logging.info('nb_1={}'.format(nb_1))

    computed_params = {'max_nb_premises': max_nb_premises,
                       'max_inputseq_len': max_inputseq_len,
                       'nb_1': nb_1,
                       'nb_0': nb_0
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

    input_answer = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='answer')
    inputs.append(input_answer)

    layers = []

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
                    pooling = keras.layers.MaxPooling1D()
                elif params['pooling'] == 'average':
                    pooling = keras.layers.AveragePooling1D()
                else:
                    raise NotImplementedError()

                conv_layer1 = pooling(conv_layer1)

                conv_layer1 = lstm(conv_layer1)
                layers.append(conv_layer1)

    elif net_arch == 'cnn':
        nb_filters = params['nb_filters']
        max_kernel_size = params['max_kernel_size']

        for kernel_size in range(1, max_kernel_size+1):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            for input in inputs:
                conv_layer1 = conv(input)

                if params['pooling'] == 'max':
                    pooling = keras.layers.GlobalMaxPooling1D()
                elif params['pooling'] == 'average':
                    pooling = keras.layers.GlobalAveragePooling1D()
                else:
                    raise NotImplementedError()

                conv_layer1 = pooling(conv_layer1)
                layers.append(conv_layer1)
    else:
        raise NotImplementedError()

    encoder_merged = keras.layers.concatenate(inputs=list(layers))
    decoder = encoder_merged

    if params['units1'] > 0:
        decoder = Dense(params['units1'], activation='relu')(decoder)

        if 'units2' in params and params['units2'] > 0:
            decoder = Dense(params['units2'], activation='relu')(decoder)

            if 'units3' in params and params['units3'] > 0:
                decoder = Dense(params['units3'], activation='relu')(decoder)

    output_dims = 2
    decoder = Dense(output_dims, activation='softmax', name='output')(decoder)

    model = Model(inputs=inputs, outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])

    return model


def generate_rows(params, computed_params, samples, batch_size, mode):
    batch_index = 0
    batch_count = 0
    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    nb_premises = computed_params['max_nb_premises']
    w1_weight = params['w1_weight']

    Xn_batch = []
    for _ in range(nb_premises + 1):
        x = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_batch.append(x)

    X_answer = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)

    inputs = {}
    for ipremise in range(nb_premises):
        inputs['premise{}'.format(ipremise)] = Xn_batch[ipremise]
    inputs['question'] = Xn_batch[nb_premises]
    inputs['answer'] = X_answer

    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    weights = np.zeros((batch_size), dtype=np.float32)
    weights.fill(1.0)

    missing_words = set()

    while True:
        for irow, sample in enumerate(samples):
            for ipremise, premise in enumerate(sample.premises):
                words = tokenizer.tokenize(premise)
                words = pad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[ipremise], batch_index, word2vec)

            words = tokenizer.tokenize(sample.question)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[nb_premises], batch_index, word2vec)

            words = tokenizer.tokenize(sample.answer)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, X_answer, batch_index, word2vec)

            if sample.label == 0:
                y_batch[batch_index, 0] = True
                weights[batch_index] = 1.0
            else:
                y_batch[batch_index, 1] = True
                weights[batch_index] = w1_weight

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield (inputs, {'output': y_batch}, weights)
                else:
                    yield inputs

                # очищаем матрицы порции для новой порции
                X_answer.fill(0)
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
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(params, computed_params, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns // batch_size)
    logging.info('max val_acc={}'.format(max(hist.history['val_acc'])))
    model.load_weights(weights_path)


def score_model(model, params, computed_params, val_samples):
    # прогоним валидационные паттерны через модель, чтобы получить f1 score.
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
    logging.info('roc_auc_score={}'.format(score))

    return f1, score


def report_model(model, params, computed_params, samples):
    # Сохраним в текстовом файле для визуальной проверки результаты валидации по всем сэмплам
    for v in generate_rows(params, computed_params, samples, len(samples), 1):
        x = v[0]
        break

    y_pred0 = model.predict(x)[:, 1]
    y_pred = (y_pred0 >= 0.5).astype(np.int)

    with io.open(os.path.join(tmp_folder, 'nn_answer_relevancy.validation.txt'), 'w', encoding='utf-8') as wrt:
        for isample, sample in enumerate(samples):
            if isample > 0:
                wrt.write(u'\n\n')

            for premise in sample.premises:
                wrt.write(u'P: {}\n'.format(premise))
            wrt.write(u'Q: {}\n'.format(sample.question))
            wrt.write(u'A: {}\n'.format(sample.answer))
            y1 = y_pred[isample]
            y2 = y_pred0[isample]
            verdict = ' (ERROR)' if y1 != y2 else ''
            wrt.write(u'true_label={} predicted_label={} (y={}){}\n'.format(sample.label, y1, y2, verdict))

# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Neural model for answer relevancy calculator')
parser.add_argument('--run_mode', type=str, default='train', choices='train gridsearch query'.split(), help='what to do: train | query | gridsearch')
parser.add_argument('--batch_size', type=int, default=250, help='batch size for neural model training')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

args = parser.parse_args()

data_folder = args.data_dir
tmp_folder = args.tmp

wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)
batch_size = args.batch_size
run_mode = args.run_mode

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_answer_relevancy.log'))

# В этих файлах будем сохранять натренированную сетку
config_path = os.path.join(tmp_folder, 'nn_answer_relevancy.config')
arch_filepath = os.path.join(tmp_folder, 'nn_answer_relevancy.arch')
weights_path = os.path.join(tmp_folder, 'nn_answer_relevancy.weights')


if run_mode == 'gridsearch':
    logging.info('Start gridsearch')

    tokenizer = Tokenizer()
    tokenizer.load()

    samples, computed_params = load_samples(input_paths, tokenizer)

    word2vec, word_dims, wordchar2vector_path = load_embeddings(tmp_folder, word2vector_path, computed_params)

    best_params = None
    best_score = -np.inf

    n0 = computed_params['nb_no']
    n1 = computed_params['nb_yes']

    params = dict()
    crossval_count = 0
    for net_arch in ['lstm(cnn)']:  #  'lstm' 'cnn' 'lstm(cnn)'
        params['net_arch'] = net_arch

        for w1_weight in [(n0/float(n0+n1)), math.sqrt((n0/float(n0+n1))), 1.0]:
            params['w1_weight'] = w1_weight

            for rnn_size in [32, 48] if net_arch in ['lstm', 'lstm(cnn)'] else [0]:
                params['rnn_size'] = rnn_size

                for nb_filters in [160, 180] if net_arch in ['cnn', 'lstm(cnn)'] else [0]:
                    params['nb_filters'] = nb_filters

                    for min_kernel_size in [1]:
                        params['min_kernel_size'] = min_kernel_size

                        for max_kernel_size in [3] if net_arch in ['cnn', 'lstm(cnn)'] else [0]:
                            params['max_kernel_size'] = max_kernel_size

                            for pooling in ['max'] if net_arch in ['cnn', 'lstm(cnn)'] else ['']:  # , 'average'
                                params['pooling'] = pooling

                                for units1 in [32]:
                                    params['units1'] = units1

                                    for units2 in [0]:
                                        params['units2'] = units2

                                        for units3 in [0]:
                                            params['units3'] = units3

                                            for batch_size in [80, 100, 120]:
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

                                                        f1_score, score = score_model(model, params, computed_params, finval_samples)
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

    #logging.info('Loading tagger...')
    #tagger = rupostagger.RuPosTagger()
    #tagger.load()

    #logging.info('Loading lemmatizer...')
    #lemmatizer = rulemma.Lemmatizer()
    #lemmatizer.load()

    samples, computed_params = load_samples(os.path.join(data_folder, 'answer_relevancy_dataset.dat'))

    word2vec, word_dims, wordchar2vector_path = load_embeddings(tmp_folder, word2vector_path, computed_params)

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
        json.dump(model_config, f, indent=4)

    params = dict()
    params['net_arch'] = 'lstm(cnn)'
    params['rnn_size'] = 64
    params['w1_weight'] = 1.0
    params['nb_filters'] = 150
    params['min_kernel_size'] = 1
    params['max_kernel_size'] = 2
    params['pooling'] = 'max'
    params['units1'] = 80
    params['units2'] = 0
    params['units3'] = 0
    params['batch_size'] = 250
    params['optimizer'] = 'nadam'

    model = create_model(params, computed_params)
    model.summary()

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=123456)
    train_model(model, params, computed_params, train_samples, val_samples)
    f1_score, logloss = score_model(model, params, computed_params, val_samples)
    report_model(model, params, computed_params, samples)


if run_mode == 'query':
    logging.info('Start run_mode==query')

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

    computed_params = dict()
    word2vec, word_dims, wordchar2vector_path = load_embeddings(tmp_folder, w2v_path, computed_params)

    tokenizer = Tokenizer()
    tokenizer.load()

    Xn_probe = []
    for i in range(max_nb_premises + 1):
        Xn_probe.append(np.zeros((1, max_inputseq_len, word_dims), dtype='float32'))
    X_answer = np.zeros((1, max_inputseq_len, word_dims), dtype='float32')

    while True:
        print('\nEnter 0 to {} premises and one question:'.format(max_nb_premises))
        premises = []
        question = None
        for ipremise in range(max_nb_premises):
            premise = utils.console_helpers.input_kbd('premise #{} :> '.format(ipremise)).strip().lower()
            if len(premise) == 0:
                break
            if premise[-1] == u'?':
                question = premise
                break

            premise = clean_phrase(premise, tokenizer)

            premises.append(premise)

        if question is None:
            question = utils.console_helpers.input_kbd('question:> ').strip().lower()
        if len(question) == 0:
            break

        question = clean_phrase(question, tokenizer)

        answer = utils.console_helpers.input_kbd('answer:> ').strip().lower()
        answer = clean_phrase(answer, tokenizer)

        # Очистим входные тензоры перед заполнением новыми данными
        X_answer.fill(0)
        for X in Xn_probe:
            X.fill(0)

        # Векторизуем входные данные - предпосылки и вопрос
        for ipremise, premise in enumerate(premises):
            words = tokenizer.tokenize(premise)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_probe[ipremise], 0, word2vec)

        words = tokenizer.tokenize(question)
        words = pad_wordseq(words, max_inputseq_len)
        vectorize_words(words, Xn_probe[max_nb_premises], 0, word2vec)

        words = tokenizer.tokenize(answer)
        words = pad_wordseq(words, max_inputseq_len)
        vectorize_words(words, X_answer, 0, word2vec)

        inputs = dict()
        for ipremise in range(max_nb_premises):
            inputs['premise{}'.format(ipremise)] = Xn_probe[ipremise]
        inputs['question'] = Xn_probe[max_nb_premises]
        inputs['answer'] = X_answer

        y_probe = model.predict(x=inputs)
        p = y_probe[0][1]
        print('p={}'.format(p))

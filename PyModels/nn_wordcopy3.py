# -*- coding: utf-8 -*-
"""
Нейросетевая модель для выбора цепочки слов, которые надо скопировать из предпосылки
для формирования ответа на вопрос.

Модель используется в проекте чат-бота https://github.com/Koziev/chatbot

Датасет должен быть предварительно сгенерирован скриптом prepare_qa_dataset.py

Кроме того, необходимо подготовить датасет с символьными встраиваниями слов, например
с помощью скрипта scripts/train_wordchar2vector.sh

Параметры обучения модели, в том числе детали архитектуры, задаются опциями
командной строки. Пример запуска обучения можно посмотреть в скрипте train_nn_wordcopy3.sh

16.03.2019 Начало переделки для введения gridsearch
"""

from __future__ import division
from __future__ import print_function

import gc
import itertools
import json
import os
import sys

import gensim
import numpy as np
import pandas as pd
import tqdm
import argparse
import logging

import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.merge import concatenate, add, multiply
from keras.layers import Lambda
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.model_selection import KFold

from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers


#padding = 'left'
padding = 'right'
PAD_WORD = u''
CONFIG_FILENAME = 'nn_wordcopy3.config'


def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова, чтобы длина стала равно n токенов"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def lpad_wordseq(words, n):
    """Слева добавляем пустые слова, чтобы длина стала равно n токенов"""
    return list(itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words))


def pad_wordseq(words, n):
    """Слева или справа добавляем пустые слова, чтобы длина стала равно n токенов"""
    if padding == 'right':
        return rpad_wordseq(words, n)
    else:
        return lpad_wordseq(words, n)


def vectorize_words(words, M, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def select_patterns(sequences, targets):
    sequences1 = []
    targets1 = []

    for seq, target in itertools.izip(sequences, targets):
        premise = seq[0]
        answer = target[0]

        all_words_mapped = True # все ли слова вопроса найдены в предпосылке?
        for word in answer:
            if word not in premise:
                all_words_mapped = False
                break

        if all_words_mapped:
            sequences1.append(seq)
            targets1.append(target)

    return sequences1, targets1


def count_words(words):
    return len(filter(lambda z: z != PAD_WORD, words))


def load_samples(input_path, wordchar2vector_path, word2vector_path):
    tokenizer = Tokenizer()
    tokenizer.load()

    max_inputseq_len = 0
    max_outputseq_len = 0  # максимальная длина ответа

    logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    logging.info('wc2v_dims={0}'.format(wc2v_dims))

    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

    logging.info('samples.count={}'.format(df.shape[0]))

    for i, record in df.iterrows():
        for phrase in [record['premise'], record['question']]:
            words = tokenizer.tokenize(phrase)
            max_inputseq_len = max(max_inputseq_len, len(words))

        phrase = record['answer']
        words = tokenizer.tokenize(phrase)
        max_outputseq_len = max(max_outputseq_len, len(words))

    computed_params = dict()
    computed_params['max_inputseq_len'] = max_inputseq_len
    computed_params['max_outputseq_len'] = max_outputseq_len

    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_outputseq_len={}'.format(max_outputseq_len))

    logging.info(u'Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path,
                                                          binary=not word2vector_path.endswith('.txt'))
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

    computed_params['word2vec'] = word2vec

    del w2v
    del wc2v
    gc.collect()

    input_data = []
    output_data = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
        premise = row['premise']
        question = row['question']
        answer = row['answer']

        premise_words = pad_wordseq(tokenizer.tokenize(premise), max_inputseq_len)
        question_words = pad_wordseq(tokenizer.tokenize(question), max_inputseq_len)

        answer_words = tokenizer.tokenize(answer)
        input_data.append((premise_words, question_words, premise, question))
        output_data.append((answer_words, answer))

    return input_data, output_data, computed_params


def create_model(params, computed_params):
    net_arch = params['net_arch']
    classifier_arch = params['classifier_arch']
    word_dims = computed_params['word_dims']
    max_inputseq_len = computed_params['max_inputseq_len']

    logging.info('Constructing the NN model {} {}...'.format(net_arch, classifier_arch))

    words_net1 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words1')
    words_net2 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words2')

    conv1 = []
    conv2 = []
    repr_size = 0

    if net_arch == 'lstm':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        rnn_size = params['rnn_size']
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        repr_size = rnn_size*2
        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)
    elif net_arch == 'lstm+cnn':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        rnn_size = params['rnn_size']
        min_kernel_size = params['min_kernel_size']
        max_kernel_size = params['max_kernel_size']
        nb_filters = params['nb_filters']

        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)
        repr_size += rnn_size*2

        # добавляем входы со сверточными слоями
        for kernel_size in range(min_kernel_size, max_kernel_size+1):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            if params['pooling'] == 'max':
                pooler = GlobalMaxPooling1D()
            elif params['pooling'] == 'average':
                pooler = GlobalAveragePooling1D()
            else:
                raise NotImplementedError()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += nb_filters
    elif net_arch == 'lstm(cnn)':
        rnn_size = params['rnn_size']
        min_kernel_size = params['min_kernel_size']
        max_kernel_size = params['max_kernel_size']
        nb_filters = params['nb_filters']

        for kernel_size in range(min_kernel_size, max_kernel_size+1):
            # сначала идут сверточные слои, образующие детекторы словосочетаний
            # и синтаксических конструкций
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            lstm = recurrent.LSTM(rnn_size, return_sequences=False)

            if params['pooling'] == 'max':
                pooler = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')
            elif params['pooling'] == 'average':
                pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')
            else:
                raise NotImplementedError()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv_layer2 = lstm(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += rnn_size
    elif net_arch == 'cnn':
        # простая сверточная архитектура.
        min_kernel_size = params['min_kernel_size']
        max_kernel_size = params['max_kernel_size']
        nb_filters = params['nb_filters']

        for kernel_size in range(min_kernel_size, max_kernel_size+1):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            if params['pooling'] == 'max':
                pooler = GlobalMaxPooling1D()
            elif params['pooling'] == 'average':
                pooler = GlobalAveragePooling1D()
            else:
                raise NotImplementedError()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += nb_filters

    # Тренируем модель, которая определяет позиции слов начала и конца цепочки.
    # Таким образом, у модели два независимых классификатора на выходе.
    output_dims = max_inputseq_len

    if classifier_arch == 'merge':
        encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2)))
        encoder_final = Dense(units=int(repr_size), activation='relu')(encoder_merged)
    elif classifier_arch == 'muladd':
        encoder1 = None
        encoder2 = None

        if len(conv1) == 1:
            encoder1 = conv1[0]
        else:
            encoder1 = keras.layers.concatenate(inputs=conv1)

        if len(conv2) == 1:
            encoder2 = conv2[0]
        else:
            encoder2 = keras.layers.concatenate(inputs=conv2)

        # сожмем вектор предложения до sent2vec_dim
        #encoder1 = sent_repr_layer(encoder1)
        #encoder2 = sent_repr_layer(encoder2)
        sent2vec_dim = repr_size

        addition = add([encoder1, encoder2])
        minus_y1 = Lambda(lambda x: -x, output_shape=(sent2vec_dim,))(encoder1)
        mul = add([encoder2, minus_y1])
        mul = multiply([mul, mul])

        #words_final = keras.layers.concatenate(inputs=[encoder1, mul, addition, encoder2])
        encoder_final = keras.layers.concatenate(inputs=[mul, addition])
    else:
        raise NotImplementedError('Unknown classifier arch: {}'.format(classifier_arch))

    decoder = Dense(params['units1'], activation='relu')(encoder_final)

    if params['units2'] > 0:
        decoder = Dense(params['units2'], activation='relu')(decoder)

        if params['units3'] > 0:
            decoder = Dense(params['units3'], activation='relu')(decoder)

    output_beg = Dense(output_dims, activation='softmax', name='output_beg')(decoder)

    decoder = Dense(params['units1'], activation='relu')(encoder_final)

    if params['units2'] > 0:
        decoder = Dense(params['units2'], activation='relu')(decoder)

        if params['units3'] > 0:
            decoder = Dense(params['units3'], activation='relu')(decoder)

    output_end = Dense(output_dims, activation='softmax', name='output_end')(decoder)

    model = Model(inputs=[words_net1, words_net2], outputs=[output_beg, output_end])
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    model.summary()

    return model


def train_model(model, params, train_input1, train_output1, val_input1, val_output1):
    nb_train_patterns = len(train_input1)
    nb_valid_patterns = len(val_input1)

    logging.info('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_output_beg_acc'

    model_checkpoint = ModelCheckpoint(weights_path,
                                       monitor=monitor_metric,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=5, verbose=1, mode='auto')

    callbacks = [model_checkpoint, early_stopping]

    batch_size = params['batch_size']
    hist = model.fit_generator(generator=generate_rows_word_copy3(train_input1, train_output1, computed_params, batch_size, 1),
                               steps_per_epoch=nb_train_patterns // batch_size,
                               epochs=100,
                               verbose=2,
                               callbacks=callbacks,
                               validation_data=generate_rows_word_copy3(val_input1, val_output1, computed_params, batch_size, 1),
                               validation_steps=nb_valid_patterns // batch_size
                               )
    val_output_beg_acc = hist.history['val_output_beg_acc']
    val_output_end_acc = hist.history['val_output_end_acc']
    logging.info('max(val_output_beg_acc)={}'.format(max(val_output_beg_acc)))
    logging.info('max(val_output_end_acc)={}'.format(max(val_output_end_acc)))
    model.load_weights(weights_path)


def generate_rows_word_copy3(sequences0, targets0, computed_params, batch_size, mode):
    """ Генератор данных в батчах для тренировки и валидации сетки """
    assert(len(sequences0) == len(targets0))
    assert(0 < batch_size < 10000)
    assert(mode in[1, 2])

    batch_index = 0
    batch_count = 0

    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    word2vec = computed_params['word2vec']
    output_dims = max_inputseq_len

    n = len(sequences0)
    shuffled_indeces = list(np.random.permutation(range(n)))
    sequences = [sequences0[i] for i in shuffled_indeces]
    targets = [targets0[i] for i in shuffled_indeces]

    X1_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)

    y1_batch = np.zeros((batch_size, output_dims), dtype=np.bool)
    y2_batch = np.zeros((batch_size, output_dims), dtype=np.bool)

    while True:
        for irow, (seq, target) in enumerate(itertools.izip(sequences, targets)):
            vectorize_words(seq[0], X1_batch, batch_index, word2vec)
            vectorize_words(seq[1], X2_batch, batch_index, word2vec)

            premise = seq[0]
            answer = target[0]

            beg_found = False
            beg_pos = 0
            end_pos = 0
            for word in answer:
                if word in premise:
                    pos = premise.index(word)
                    if beg_found == False:
                        beg_found = True
                        beg_pos = pos
                    end_pos = max(end_pos, pos)
                else:
                    break

            assert(end_pos >= beg_pos)

            y1_batch[batch_index, beg_pos] = True
            y2_batch[batch_index, end_pos] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                x = {'input_words1': X1_batch, 'input_words2': X2_batch}
                if mode == 1:
                    yield (x, {'output_beg': y1_batch, 'output_end': y2_batch})
                else:
                    yield x

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                y1_batch.fill(0)
                y2_batch.fill(0)
                batch_index = 0


def score_model(model, inputs, outputs, params):
    batch_size = params['batch_size']
    n = len(inputs) // batch_size
    accs1 = []
    accs2 = []
    for ibatch, batch_data in enumerate(generate_rows_word_copy3(inputs, outputs, computed_params, batch_size, 1)):
        x_data = batch_data[0]
        y_data = batch_data[1]
        y_true1 = np.argmax(y_data['output_beg'], axis=-1)
        y_true2 = np.argmax(y_data['output_end'], axis=-1)

        y_pred12 = model.predict(x_data, verbose=0)
        y_pred1 = y_pred12[0]
        y_pred2 = y_pred12[1]
        y_pred1 = np.argmax(y_pred1, axis=-1)
        y_pred2 = np.argmax(y_pred2, axis=-1)

        acc1 = sklearn.metrics.accuracy_score(y_true=y_true1, y_pred=y_pred1)
        accs1.append(acc1)

        acc2 = sklearn.metrics.accuracy_score(y_true=y_true2, y_pred=y_pred2)
        accs2.append(acc2)

        if ibatch == n-1:
            break

    acc1 = np.mean(accs1)
    acc2 = np.mean(accs2)
    return 0.5 * (acc1 + acc2)

# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Neural model for word copy model for answer generation')
parser.add_argument('--run_mode', type=str, default='train', choices='gridsearch train query'.split(), help='what to do: train | query | gridsearch')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/premise_question_answer.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')

args = parser.parse_args()
input_path = args.input
tmp_folder = args.tmp
run_mode = args.run_mode
batch_size = args.batch_size
wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_wordcopy3.log'))

# В этих файлах будем сохранять натренированную сетку
arch_filepath = os.path.join(tmp_folder, 'nn_wordcopy3.arch')
weights_path = os.path.join(tmp_folder, 'nn_wordcopy3.weights')


if run_mode == 'gridsearch':
    logging.info('Start gridsearch')

    input_data, output_data, computed_params = load_samples(input_path, wordchar2vector_path, word2vector_path)

    best_params = None
    best_score = -np.inf

    params = dict()
    crossval_count = 0
    for net_arch in ['lstm', 'lstm(cnn)']:  # ,
        params['net_arch'] = net_arch

        for classifier_arch in ['merge']:
            params['classifier_arch'] = classifier_arch

            for rnn_size in [64]:
                params['rnn_size'] = rnn_size

                for pooling in ['max', 'average'] if net_arch == 'lstm(cnn)' else ['']:
                    params['pooling'] = pooling

                    for nb_filters in [32, 48, 64] if net_arch == 'lstm(cnn)' else [0]:
                        params['nb_filters'] = nb_filters

                        for min_kernel_size in [1] if net_arch == 'lstm(cnn)' else [0]:
                            params['min_kernel_size'] = min_kernel_size

                            for max_kernel_size in [2, 3] if net_arch == 'lstm(cnn)' else [0]:
                                params['max_kernel_size'] = max_kernel_size

                                for units1 in [48, 64, 80]:
                                    params['units1'] = units1

                                    for units2 in [32]:
                                        params['units2'] = units2

                                        for units3 in [0]:
                                            params['units3'] = units3

                                            for batch_size in [250]:
                                                params['batch_size'] = batch_size

                                                for optimizer in ['nadam']:
                                                    params['optimizer'] = optimizer

                                                    crossval_count += 1
                                                    logging.info('Crossvalidation #{} for {}'.format(crossval_count,
                                                                                                     get_params_str(
                                                                                                         params)))

                                                    model = create_model(params, computed_params)

                                                    kf = KFold(n_splits=3)
                                                    scores = []
                                                    for ifold, (train_index, val_index) in enumerate(kf.split(input_data)):
                                                        logging.info('KFold[{}]'.format(ifold))

                                                        train_inputs = [input_data[i] for i in train_index]
                                                        train_outputs = [output_data[i] for i in train_index]
                                                        train_inputs, train_outputs = select_patterns(train_inputs,
                                                                                                      train_outputs)

                                                        val12_inputs = [input_data[i] for i in val_index]
                                                        val12_outputs = [output_data[i] for i in val_index]

                                                        val12_inputs, val12_outputs = select_patterns(val12_inputs,
                                                                                                      val12_outputs)

                                                        val_inputs, finval_inputs,\
                                                        val_outputs, finval_outputs = train_test_split(val12_inputs,
                                                                                                       val12_outputs,
                                                                                                       test_size=0.5,
                                                                                                       random_state=123456789)

                                                        train_model(model, params,
                                                                    train_inputs, train_outputs,
                                                                    val_inputs, val_outputs)

                                                        score = score_model(model, finval_inputs, finval_outputs, params)
                                                        logging.info('model validation score={}'.format(score))
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

    input_data, output_data, computed_params = load_samples(input_path, wordchar2vector_path, word2vector_path)

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'engine': 'nn',
        'max_inputseq_len': computed_params['max_inputseq_len'],
        'max_outputseq_len': computed_params['max_outputseq_len'],
        'w2v_path': word2vector_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'padding': padding,
        'model_folder': tmp_folder,
        'word_dims': computed_params['word_dims']
    }

    with open(os.path.join(tmp_folder, CONFIG_FILENAME), 'w') as f:
        json.dump(model_config, f, indent=4)

    params = dict()
    params['net_arch'] = 'lstm'
    params['classifier_arch'] = 'merge'

    params['rnn_size'] = 100  #64

    if params['net_arch'] == 'lstm(cnn)':
        params['pooling'] = 'max'
        params['nb_filters'] = 128
        params['min_kernel_size'] = 1
        params['max_kernel_size'] = 3

    params['units1'] = 64
    params['units2'] = 0  #32
    params['units3'] = 0
    params['batch_size'] = 250
    params['optimizer'] = 'nadam'

    model = create_model(params, computed_params)

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    SEED = 123456
    TEST_SHARE = 0.2
    train_input, val_input, train_output, val_output = train_test_split(input_data,
                                                                        output_data,
                                                                        test_size=TEST_SHARE,
                                                                        random_state=SEED)

    train_input1, train_output1 = select_patterns(train_input, train_output)
    val_input1, val_output1 = select_patterns(val_input, val_output)

    train_model(model, params, train_input1, train_output1, val_input1, val_output1)

    acc = score_model(model, val_input1, val_output1, params)
    logging.info('Final model score={}'.format(acc))

if run_mode == 'query':
    # Ручная проверка работы натренированной модели.

    with open(os.path.join(tmp_folder, CONFIG_FILENAME), 'r') as f:
        model_config = json.load(f)

    max_inputseq_len = model_config['max_inputseq_len']
    max_outputseq_len = model_config['max_outputseq_len']
    word2vector_path = model_config['w2v_path']
    wordchar2vector_path = model_config['wordchar2vector_path']
    word_dims = model_config['word_dims']
    padding = model_config['padding']

    tokenizer = Tokenizer()
    tokenizer.load()

    print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    print('wc2v_dims={0}'.format(wc2v_dims))

    print('Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path,
                                                          binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))

    word_dims = w2v_dims + wc2v_dims

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros(word_dims)
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    del w2v
    del wc2v
    gc.collect()

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    X1_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)

    while True:
        print('\nEnter two phrases:')
        phrase1 = utils.console_helpers.input_kbd('premise :> ').strip().lower()
        if len(phrase1) == 0:
            break

        phrase2 = utils.console_helpers.input_kbd('question:> ').strip().lower()
        if len(phrase2) == 0:
            break

        words1 = tokenizer.tokenize(phrase1)
        words2 = tokenizer.tokenize(phrase2)

        words1 = pad_wordseq(words1, max_inputseq_len)
        words2 = pad_wordseq(words2, max_inputseq_len)

        X1_probe.fill(0)
        X2_probe.fill(0)

        vectorize_words(words1, X1_probe, 0, word2vec)
        vectorize_words(words2, X2_probe, 0, word2vec)

        y_probe = model.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})

        y1 = y_probe[0][0]
        y2 = y_probe[1][0]

        pos_beg = np.argmax(y1)
        pos_end = np.argmax(y2)

        print('pos_beg={} pos_end={}'.format(pos_beg, pos_end))

        answer_words = words1[pos_beg:pos_end + 1]
        print(u'{}'.format(u' '.join(answer_words)))

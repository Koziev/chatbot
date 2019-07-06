# -*- coding: utf-8 -*-
"""
Модель для разметки токенов в строке (общая задача NER)
Должна быть использована в NLU модуле чатбота.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import gensim
import numpy as np
import io
import argparse
import random
import logging
import collections

import keras
import keras.initializers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPool1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed

import keras_contrib
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import keras_contrib.optimizers
from keras_contrib.optimizers import FTML

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics

from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers


class Sample:
    def __init__(self, phrase, entity, phrase_tokens, token_labels):
        self.phrase = phrase
        self.entity = entity
        self.phrase_tokens = phrase_tokens
        self.token_labels = token_labels

    def write(self, wrt):
        for premise in self.premises:
            wrt.write(u'T: {}\n'.format(u' '.join(premise)))
        wrt.write(u'Q: {}\n'.format(u' '.join(self.question)))
        wrt.write(u'A: {}\n'.format(u' '.join(self.answer)))


def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


def count_words(words):
    """ Возвращает кол-во непустых строк в последовательности words """
    return len(filter(lambda z: z != PAD_WORD,words))


def lpad_wordseq(words, n):
    """ Слева добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """ Справа добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain( words, itertools.repeat(PAD_WORD, n-len(words))))


def pad_wordseq(words, n, padding):
    if padding == 'right':
        return rpad_wordseq(words, n)
    else:
        return lpad_wordseq(words, n)




def map_value(phrase_tokens, value_tokens):
    labels = np.zeros(len(phrase_tokens), dtype=np.int)
    ntoken = len(value_tokens)
    try:
        if ntoken > 0:
            mapped = False
            if value_tokens[0] in phrase_tokens:
                for i0, phrase_token in enumerate(phrase_tokens):
                    if phrase_token == value_tokens[0]:
                        # первый токен совпал
                        # проверим совпадение оставшихся токенов
                        tail_matched = True
                        if len(value_tokens) == 1:
                            labels[i0] = 1
                            mapped = True
                        else:
                            for i1, value_token in enumerate(value_tokens[1:], start=1):
                                if i1 < ntoken:
                                    if value_token != phrase_tokens[i0 + i1]:
                                        tail_matched = False
                                        break
                                else:
                                    tail_matched = False
                                    break

                            if tail_matched:
                                labels[i0:i0+i1+1] = 1
                                mapped = True
            if not mapped:
                msg = u'NOT MAPPED ERROR: phrase_tokens={} value_tokens={}'.format(u' '.join(phrase_tokens), u' '.join(value_tokens))
                raise RuntimeError(msg)
    except Exception as ex:
        print(u'Error occured for phrase="{}" and entity="{}":\n{}'.format(u' '.join(phrase_tokens), u' '.join(value_tokens), ex))
        return None

    return labels

def load_samples(input_path):
    computed_params = dict()

    logging.info(u'Loading samples from {}'.format(input_path))

    # Для каждого класса извлекаемых сущностей получаем отдельный набор сэмплов
    entity2samples = dict()

    max_inputseq_len = 0
    tokenizer = Tokenizer()
    tokenizer.load()
    all_value_tokens = set()

    with io.open(input_path, 'r', encoding='utf-8') as rdr:
        current_entity = None
        for line in rdr:
            line = line.strip()
            if line and not line.startswith('#'):  # пропускаем комментарии и пустые строки
                if line.startswith('entity='):
                    current_entity = line.split('=')[1]
                    if current_entity not in entity2samples:
                        entity2samples[current_entity] = []
                else:
                    tx = line.split('|')
                    phrase = tx[0].strip()
                    value = u'' if len(tx) == 1 else tx[1].strip()

                    if phrase.endswith(u'.'):
                        phrase = phrase[:-1]
                    phrase = phrase.replace(u'?', u'').replace(u'!', u'')

                    if value.endswith(u'.'):
                        value = value[:-1]

                    phrase_tokens = tokenizer.tokenize(phrase)
                    value_tokens = tokenizer.tokenize(value)

                    all_value_tokens.update(value_tokens)
                    token_labels = map_value(phrase_tokens, value_tokens)
                    if token_labels is not None:
                        sample = Sample(phrase, value, phrase_tokens, token_labels)
                        entity2samples[current_entity].append(sample)
                        max_inputseq_len = max(max_inputseq_len, len(phrase_tokens))

    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    computed_params['max_inputseq_len'] = max_inputseq_len
    computed_params['value_tokens'] = all_value_tokens

    logging.info('Count of samples per entity:')
    for entity, samples in sorted(entity2samples.items(), key=lambda samples: len(samples)):
        logging.info(u'entity={} count={}'.format(entity, len(samples)))

    return entity2samples, computed_params


def load_embeddings(tmp_folder, word2vector_path, computed_params):
    wordchar2vector_path = os.path.join(tmp_folder,'wordchar2vector.dat')
    logging.info('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    logging.info('wc2v_dims={0}'.format(wc2v_dims))

    logging.info('Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    logging.info('w2v_dims={0}'.format(w2v_dims))

    word_dims = w2v_dims + wc2v_dims   # + 1  # 1 компонент для бинарного признака "известный токен для значения entity"
    computed_params['word_dims'] = word_dims

    value_tokens = computed_params['value_tokens']

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros(word_dims)
        v[w2v_dims:w2v_dims+wc2v_dims] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        #v[word_dims-1] = word in value_tokens
        word2vec[word] = v

    computed_params['word2vec'] = word2vec

    del w2v
    #del wc2v
    gc.collect()


def vectorize_samples(samples, params, computed_params):
    padding = params['padding']
    nb_samples = len(samples)
    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    w2v = computed_params['word2vec']

    X_data = np.zeros((nb_samples, max_inputseq_len, word_dims), dtype=np.float32)
    y_data = np.zeros((nb_samples, max_inputseq_len, 2), dtype=np.bool)

    for isample, sample in enumerate(samples):
        words = pad_wordseq(sample.phrase_tokens, max_inputseq_len, padding)
        vectorize_words(words, X_data, isample, w2v)

        # По умолчанию считаем, что для всех токенов, включая заполнители, на выходе
        # будет label=0. Поэтому выставляем нулевой элемент каждой пары в 1.
        y_data[isample, :, 0] = 1

        for itoken, token_label in enumerate(sample.token_labels):
            y_data[isample, itoken, 1-token_label] = 0
            y_data[isample, itoken, token_label] = 1

    return X_data, y_data



def create_model(params, computed_params):
    input = Input(shape=(computed_params['max_inputseq_len'], computed_params['word_dims'],),
                  dtype='float32', name='input')

    if params['optimizer'] == 'ftml':
        opt = keras_contrib.optimizers.FTML()
    else:
        opt = params['optimizer']

    if params['net_arch'] == 'crf':
        net = input

        for _ in range(params['nb_rnn']):
            net = Bidirectional(recurrent.LSTM(units=params['rnn_units1'],
                                               dropout=params['dropout_rate'],
                                               kernel_initializer='glorot_normal',
                                               return_sequences=True))(net)

        net = CRF(units=2, sparse_target=False)(net)
        model = Model(inputs=[input], outputs=net)
        model.compile(loss=crf_loss, optimizer=opt, metrics=[crf_viterbi_accuracy])
    elif params['net_arch'] == 'lstm':
        net = Bidirectional(recurrent.LSTM(units=params['rnn_units1'],
                                           dropout=params['dropout_rate'],
                                           return_sequences=False))(input)

        for _ in range(params['nb_dense1']):
            net = Dense(units=params['rnn_units1'], activation=params['activation1'])(net)

        decoder = RepeatVector(computed_params['max_inputseq_len'])(net)
        decoder = recurrent.LSTM(params['rnn_units2'], return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(units=2, activation='softmax'), name='output')(decoder)

        model = Model(inputs=[input], outputs=decoder)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
    elif params['net_arch'] == 'lstm(cnn)':
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

            lstm = recurrent.LSTM(params['rnn_units1'], dropout=params['dropout_rate'], return_sequences=False)

            conv_layer1 = conv(input)

            if params['pooling'] == 'max':
                pooling = keras.layers.MaxPooling1D()
            elif params['pooling'] == 'average':
                pooling = keras.layers.AveragePooling1D()
            else:
                raise NotImplementedError()

            conv_layer1 = pooling(conv_layer1)
            conv_layer1 = lstm(conv_layer1)

            decoder = conv_layer1
            for _ in range(params['nb_dense1']):
                decoder = Dense(units=params['rnn_units1'], activation=params['activation1'])(decoder)

            decoder = RepeatVector(computed_params['max_inputseq_len'])(decoder)
            decoder = recurrent.LSTM(params['rnn_units2'], return_sequences=True)(decoder)
            decoder = TimeDistributed(Dense(units=2, activation='softmax'), name='output')(decoder)



        model = Model(inputs=[input], outputs=decoder)
        model.compile(loss='categorical_crossentropy', optimizer=opt)


    #model.summary()
    return model


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate( words ):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def train_model(model, X_train, y_train, X_val, y_val, params, computed_params):
    weights_path = os.path.join(tmp_folder, 'nn_entity_extractor.weights')
    if params['net_arch'] == 'crf':
        monitor_metric = 'val_crf_viterbi_accuracy'
    else:
        monitor_metric = 'val_loss'

    model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    callbacks = [model_checkpoint, early_stopping]

    model.fit(x=X_train, y=y_train,
              validation_data=(X_val, y_val),
              epochs=1000,
              batch_size=params['batch_size'],
              verbose=2,
              callbacks=callbacks)

    model.load_weights(weights_path)

    y_pred = model.predict(X_val)

    # оценка точности per instance
    nb_total = len(y_val)
    nb_good = sum(np.array_equal(y_pred[i, :], y_val[i, :]) for i in range(nb_total))
    acc_perinstance = nb_good / float(nb_total)

    # оценка точности per token
    # (!) получается слишком оптимистичная оценка из-за большого количества заполнителей
    yy_pred = np.argmax(y_pred, axis=-1)
    yy_val = np.argmax(y_val,  axis=-1)
    nb_total = yy_pred.shape[0] * yy_pred.shape[1]
    nb_good = 0
    for irow in range(yy_pred.shape[0]):
        n1 = np.sum(np.equal(yy_pred[irow], yy_val[irow]))
        nb_good += n1
    acc_pertoken = nb_good / float(nb_total)

    return acc_pertoken, acc_perinstance


def report_model(model, samples, params, computed_params, outpath):
    X_data, y_data = vectorize_samples(samples, params, computed_params)
    max_inputseq_len = computed_params['max_inputseq_len']
    padding = params['padding']
    y_pred = model.predict(X_data, verbose=0)
    with io.open(outpath, 'w', encoding='utf-8') as wrt:
        for isample, sample in enumerate(samples):
            y = y_pred[isample]
            yy_pred = np.argmax(y, axis=-1)
            yy_true = sample.token_labels

            wrt.write(u'phrase={}\n'.format(u' '.join(sample.phrase_tokens)))

            true_words = [word for word, label in zip(sample.phrase_tokens, yy_true) if label == 1]
            wrt.write(u'true={}\n'.format(u' '.join(true_words)))

            padded_tokens = pad_wordseq(sample.phrase_tokens, max_inputseq_len, padding)
            selected_words = [word for word, label in zip(padded_tokens, yy_pred) if label == 1]
            wrt.write(u'predicted={}\n'.format(u' '.join(selected_words)))
            wrt.write(u'\n\n')


PAD_WORD = u''


class GridGenerator(object):
    def __init__(self):
        pass

    def generate(self):
        params = dict()

        for batch_size in [50, 100, 150]:
            params['batch_size'] = batch_size

            for opt in ['nadam']:
                params['optimizer'] = opt

                for net_arch in ['crf']:  # 'lstm', 'lstm(cnn)', 'crf'
                    params['net_arch'] = net_arch

                    if net_arch == 'lstm(cnn)':
                        for rnn_units1 in [50, 100, 150]:
                            params['rnn_units1'] = rnn_units1

                            for rnn_units2 in [50]:
                                params['rnn_units2'] = rnn_units2

                                for dropout_rate in [0.0]:
                                    params['dropout_rate'] = dropout_rate

                                    for nb_dense1 in [1, 2]:
                                        params['nb_dense1'] = nb_dense1

                                        for activation1 in ['sigmoid', 'relu']:
                                            params['activation1'] = activation1

                                            for nb_filters in [20, 50]:
                                                params['nb_filters'] = nb_filters

                                                for max_kernel_size in [2, 3]:
                                                    params['max_kernel_size'] = max_kernel_size

                                                    for pooling in ['max', 'average']:
                                                        params['pooling'] = pooling

                                                        for padding in ['left', 'right']:
                                                            params['padding'] = padding

                                                            yield params

                    if net_arch == 'lstm':
                        for rnn_units1 in [50, 100, 150]:
                            params['rnn_units1'] = rnn_units1

                            for rnn_units2 in [50]:
                                params['rnn_units2'] = rnn_units2

                                for dropout_rate in [0.0]:
                                    params['dropout_rate'] = dropout_rate

                                    for nb_dense1 in [1, 2]:
                                        params['nb_dense1'] = nb_dense1

                                        for activation1 in ['sigmoid', 'relu']:
                                            params['activation1'] = activation1

                                            for padding in ['left', 'right']:
                                                params['padding'] = padding

                                                yield params

                    elif net_arch == 'crf':
                        for nb_rnn in [1, 2]:
                            params['nb_rnn'] = nb_rnn

                            for rnn_units1 in [200, 250, 300]:
                                params['rnn_units1'] = rnn_units1

                                for dropout_rate in [0.0]:
                                    params['dropout_rate'] = dropout_rate

                                    for padding in ['left', 'right']:
                                        params['padding'] = padding

                                        yield params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural model for NER')
    parser.add_argument('--run_mode', type=str, default='gridsearch', choices='train query gridsearch'.split(), help='what to do: train | query | gridsearch')
    parser.add_argument('--batch_size', type=int, default=250, help='batch size for neural model training')
    parser.add_argument('--input', type=str, default='../data/entity_extraction.txt', help='path to input dataset')
    parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
    parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
    parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')
    parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

    args = parser.parse_args()

    data_folder = args.data_dir
    input_path = args.input
    tmp_folder = args.tmp

    # настраиваем логирование в файл
    utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_entity_extractor.log'))

    wordchar2vector_path = args.wordchar2vector
    word2vector_path = os.path.expanduser(args.word2vector)
    run_mode = args.run_mode

    # В этих файлах будем сохранять натренированную сетку
    arch_filepath = os.path.join(tmp_folder, 'nn_entity_extractor.arch')
    weights_path = os.path.join(tmp_folder, 'nn_entity_extractor.weights')
    config_path = os.path.join(tmp_folder, 'nn_entity_extractor.config')

    if run_mode == 'gridsearch':
        logging.info('Start gridsearch')
        entity2samples, computed_params = load_samples(input_path)
        load_embeddings(tmp_folder, word2vector_path, computed_params)

        # ищем entity с максимальным кол-вом образцов разбора
        entity, samples = sorted(entity2samples.items(), key=lambda z: -len(z[1]))[0]
        logging.info(u'Do gridsearch on {} samples for "{}" entity'.format(len(samples), entity))

        best_params = None
        best_score = -np.inf
        crossval_count = 0

        grid = GridGenerator()
        for params in grid.generate():
            crossval_count += 1
            logging.info('Crossvalidation #{} for {}'.format(crossval_count, get_params_str(params)))

            X_data, y_data = vectorize_samples(samples, params, computed_params)

            kf = KFold(n_splits=9)
            scores = []
            for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
                logging.info('KFold[{}]'.format(ifold))
                train_samples = [samples[i] for i in train_index]
                #val12_samples = [samples[i] for i in val_index]
                val_samples = [samples[i] for i in val_index]

                #SEED = 123456
                #TEST_SHARE = 0.2
                #val_samples, finval_samples = train_test_split(val12_samples, test_size=0.5, random_state=SEED)

                X_train, y_train = vectorize_samples(train_samples, params, computed_params)
                X_val, y_val = vectorize_samples(val_samples, params, computed_params)
                #X_finval, y_vinval = vectorize_samples(finval_samples, computed_params)

                model = create_model(params, computed_params)
                acc_pertoken, acc_perinstance = train_model(model, X_train, y_train, X_val, y_val, params, computed_params)
                logging.info('KFold[{}] acc_pertoken={} acc_perinstance={}'.format(ifold, acc_pertoken, acc_perinstance))
                score = acc_pertoken
                #score = score_model(model, finval_samples, params, computed_params)
                scores.append(score)

            score = np.mean(scores)
            score_std = np.std(scores)
            logging.info('Crossvalidation #{} score={} std={}'.format(crossval_count, score, score_std))
            if score > best_score:
                best_params = params.copy()
                best_score = score
                logging.info('!!! NEW BEST score={} params={}'.format(best_score, get_params_str(best_params)))
            else:
                logging.info('No improvement over current best_score={}'.format(best_score))

        logging.info(
            'Grid search complete, best_score={} best_params={}'.format(best_score, get_params_str(best_params)))

    if run_mode == 'train':
        logging.info('Start training...')

        entity2samples, computed_params = load_samples(input_path)

        # пока будет работать только с entity=когда
        entity, samples = sorted(entity2samples.items(), key=lambda z: -len(z[1]))[0]
        logging.info(u'Do training on {} samples for "{}" entity'.format(len(samples), entity))

        load_embeddings(tmp_folder, word2vector_path, computed_params)

        params = dict()
        params['net_arch'] = 'crf'
        params['nb_rnn'] = 1
        params['rnn_units1'] = 300
        params['dropout_rate'] = 0.0
        params['optimizer'] = 'nadam'
        params['batch_size'] = 100
        params['padding'] = 'right'

        # сохраним конфиг модели, чтобы ее использовать в чат-боте
        model_config = {
                        'engine': 'nn',
                        'max_inputseq_len': computed_params['max_inputseq_len'],
                        'word2vector_path': word2vector_path,
                        'wordchar2vector_path': wordchar2vector_path,
                        'PAD_WORD': PAD_WORD,
                        'model_folder': tmp_folder,
                        'word_dims': computed_params['word_dims'],
                        'padding': params['padding']
                       }

        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=4)

        model = create_model(params, computed_params)

        with open(arch_filepath, 'w') as f:
            f.write(model.to_json())

        #keras.utils.plot_model(model,
        #                       to_file=os.path.join(tmp_folder, 'nn_answer_length.arch.png'),
        #                       show_shapes=False,
        #                       show_layer_names=True)

        SEED = 123456
        TEST_SHARE = 0.2
        train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)

        X_train, y_train = vectorize_samples(train_samples, params, computed_params)
        X_val, y_val = vectorize_samples(val_samples, params, computed_params)

        acc_pertoken, acc_perinstance = train_model(model, X_train, y_train, X_val, y_val, params, computed_params)
        logging.info('acc_pertoken={}'.format(acc_pertoken))
        logging.info('acc_perinstance={}'.format(acc_perinstance))

        #val_acc = score_model(model, val_samples, params, computed_params)
        #logging.info('Final val_acc={}'.format(val_acc))

        # Для контроля прогоним через модель все сэмплы и сохраним результаты в текстовый файл
        report_model(model, samples, params, computed_params, os.path.join(tmp_folder, 'nn_entity_extractor.validation.txt'))

    if run_mode == 'query':
        logging.info('Start run_mode==query')

        with open(config_path, 'r') as f:
            model_config = json.load(f)
            max_inputseq_len = model_config['max_inputseq_len']
            w2v_path = model_config['word2vector_path']
            wordchar2vector_path = model_config['wordchar2vector_path']
            word_dims = model_config['word_dims']
            padding = model_config['padding']

        with open(arch_filepath, 'r') as f:
            model = model_from_json(f.read(), {'CRF': CRF})

        model.load_weights(weights_path)

        computed_params = dict()
        load_embeddings(tmp_folder, w2v_path, computed_params)
        word_dims = computed_params['word_dims']
        word2vec = computed_params['word2vec']

        tokenizer = Tokenizer()
        tokenizer.load()

        X_probe = np.zeros((1, max_inputseq_len, word_dims), dtype='float32')

        while True:
            phrase = utils.console_helpers.input_kbd(':> ').strip().lower()
            if len(phrase) == 0:
                break

            # Очистим входные тензоры перед заполнением новыми данными
            X_probe.fill(0)

            words = tokenizer.tokenize(phrase)
            words = pad_wordseq(words, max_inputseq_len, padding)
            vectorize_words(words, X_probe, 0, word2vec)

            inputs = dict()
            inputs['input'] = X_probe

            y = model.predict(x=inputs)[0]
            predicted_labels = np.argmax(y, axis=-1)

            selected_words = [word for word, label in zip(words, predicted_labels) if label == 1]
            print(u'prediction="{}"'.format(u' '.join(selected_words)))

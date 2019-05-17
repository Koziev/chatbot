# -*- coding: utf-8 -*-
"""
Модель для определения числа слов в ответе по заданной предпосылке и вопросу.
Модель предназначена для использования в чатботе https://github.com/Koziev/chatbot
Датасет должен быть предварительно сгенерирован скриптом prepare_qa_dataset.py
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
import tqdm
import argparse
import random
import logging
from collections import Counter

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPool1D
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


class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer

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


def pad_wordseq(words, n):
    """ Слева добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """ Справа добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain( words, itertools.repeat(PAD_WORD, n-len(words))))


def load_samples(input_path, max_nb_samples=1000000000):
    computed_params = dict()

    logging.info(u'Loading samples from {}'.format(input_path))
    samples = []
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    max_inputseq_len = 0
    tokenizer = Tokenizer()
    tokenizer.load()

    with io.open(input_path, 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if len(line) == 0:
                if len(lines) > 0:
                    premises = [tokenizer.tokenize(premise) for premise in lines[:-2]]
                    question = tokenizer.tokenize(lines[-2])
                    answer = tokenizer.tokenize(lines[-1])
                    sample = Sample(premises, question, answer)
                    samples.append(sample)

                    max_nb_premises = max(max_nb_premises, len(premises))

                    for phrase in lines:
                        words = tokenizer.tokenize(phrase)
                        max_inputseq_len = max(max_inputseq_len, len(words))

                    lines = []

            else:
                lines.append(line)

    logging.info('samples.count={}'.format(len(samples)))
    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_nb_premises={}'.format(max_nb_premises))

    computed_params['max_inputseq_len'] = max_inputseq_len
    computed_params['max_nb_premises'] = max_nb_premises

    # Гистограмма длин ответов
    answerlen2count = Counter()
    answerlen2count.update(len(sample.answer) for sample in samples)

    logging.info('max_inputseq_len={}'.format(max_inputseq_len))

    max_outputseq_len = max(len(sample.answer) for sample in samples)
    logging.info('max_outputseq_len={}'.format(max_outputseq_len))
    computed_params['max_outputseq_len'] = max_outputseq_len

    logging.info('Count of answers by number of words:')
    for l, n in sorted(answerlen2count.iteritems(), key=lambda z: z[0]):
        logging.info('number of words={:3} count={}'.format(l, n))

    if len(samples) > max_nb_samples:
        logging.info('Truncating samples list to {} items'.format(max_nb_samples))
        samples = sorted(samples, key=lambda _: random.random())[:max_nb_samples]

    return samples, computed_params


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

    word_dims = w2v_dims+wc2v_dims
    computed_params['word_dims'] = word_dims

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros( word_dims )
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    computed_params['word2vec'] = word2vec

    del w2v
    #del wc2v
    gc.collect()


def create_model(params, computed_params):
    logging.info('Constructing the NN model...')

    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    max_outputseq_len = computed_params['max_outputseq_len']
    max_nb_premises = computed_params['max_nb_premises']

    inputs = []
    input_question = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='question')
    inputs.append(input_question)

    for ipremise in range(max_nb_premises):
        input_premise = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='premise{}'.format(ipremise))
        inputs.append(input_premise)

    layers = []

    net_arch = params['net_arch']

    if net_arch == 'lstm':
        # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения. Этот слой общий для всех входных предложений.
        rnn_size = params['rnn_size']
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        for input in inputs:
            encoder_rnn = shared_words_rnn(input)
            layers.append(encoder_rnn)
    elif net_arch == 'lstm(cnn)':
        nb_filters = params['nb_filters']
        rnn_size = params['rnn_size']
        for kernel_size in range(1, 4):
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
                conv_layer1 = keras.layers.MaxPooling1D(pool_size=kernel_size,
                                                        strides=None,
                                                        padding='valid')(conv_layer1)
                conv_layer1 = lstm(conv_layer1)
                layers.append(conv_layer1)

    encoder_merged = keras.layers.concatenate(inputs=list(layers))

    # финальный классификатор определяет длину ответа
    output_dims = max_outputseq_len
    decoder = encoder_merged

    if 'units1' in params and params['units1'] > 0:
        decoder = Dense(units=params['units1'], activation='relu')(decoder)

    if 'units2' in params and params['units2'] > 0:
        decoder = Dense(params['units2'], activation='relu')(decoder)

    if 'units3' in params and params['units3'] > 0:
        decoder = Dense(params['units3'], activation='relu')(decoder)

    decoder = Dense(output_dims, activation='softmax', name='output')(decoder)

    model = Model(inputs=inputs, outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])

    return model


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate( words ):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def generate_rows(computed_params, samples, batch_size, mode):
    batch_index = 0
    batch_count = 0

    max_nb_premises = computed_params['max_nb_premises']
    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    word2vec = computed_params['word2vec']
    max_outputseq_len = computed_params['max_outputseq_len']

    Xn_batch = []
    for _ in range(max_nb_premises+1):
        x = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_batch.append(x)

    inputs = {}
    for ipremise in range(max_nb_premises):
        inputs['premise{}'.format(ipremise)] = Xn_batch[ipremise]
    inputs['question'] = Xn_batch[max_nb_premises]

    y_batch = np.zeros((batch_size, max_outputseq_len), dtype=np.bool)

    batch_samples = []

    weights = np.zeros((batch_size))
    weights.fill(1.0)

    while True:
        for irow, sample in enumerate(samples):
            for ipremise, premise in enumerate(sample.premises):
                words = premise
                words = pad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[ipremise], batch_index, word2vec)

            words = sample.question
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[max_nb_premises], batch_index, word2vec)

            # начало отладки
            #if len(sample.answer) == 13:
            #    print('DEBUG@289')
            #    print(u'question={}'.format(u' '.join(sample.question)))
            #    print(u'answer={}'.format(u' '.join(sample.answer)))
            #    exit(0)
            # конец отладки

            y_batch[batch_index, len(sample.answer)-1] = True

            if mode == 2:
                batch_samples.append(sample)

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 2:
                    yield (inputs, {'output': y_batch}, weights, batch_samples)
                elif mode == 1:
                    yield (inputs, {'output': y_batch}, weights)
                else:
                    yield inputs

                # очищаем матрицы порции для новой порции
                for x in Xn_batch:
                    x.fill(0)
                y_batch.fill(0)
                batch_samples = []
                batch_index = 0


def train_model(model, train_samples, val_samples, params, computed_params):
    batch_size = params['batch_size']
    nb_train_patterns = len(train_samples)
    nb_valid_patterns = len(val_samples)
    logging.info('Start training using {} patterns, {} validation samples...'.format(nb_train_patterns,
                                                                                     nb_valid_patterns))

    monitor_metric = 'val_acc'

    model_checkpoint = ModelCheckpoint(weights_path,
                                       monitor=monitor_metric,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.00001)

    callbacks = [reduce_lr, model_checkpoint, early_stopping]

    hist = model.fit_generator(generator=generate_rows(computed_params, train_samples, batch_size, 1),
                               steps_per_epoch=nb_train_patterns // batch_size,
                               epochs=1000,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(computed_params, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns // batch_size
                               )

    model.load_weights(weights_path)


def score_model(model, samples, params, computed_params):
    batch_size = params['batch_size']
    nb_steps = len(samples) // batch_size
    steps = 0

    sum_acc = 0.0
    denom_acc = 0

    for batch in generate_rows(computed_params, samples, batch_size, 1):
        X = batch[0]
        y = batch[1]['output']
        y = np.argmax(y, axis=-1)

        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=-1)

        acc = sklearn.metrics.accuracy_score(y_true=y, y_pred=y_pred)
        sum_acc += acc
        denom_acc += 1

        steps += 1
        if steps == nb_steps:
            break

    final_acc = sum_acc / denom_acc
    return final_acc


def report_model(model, samples, params, output_path):
    with io.open(output_path, 'w', encoding='utf-8') as wrt:
        batch_size = params['batch_size']
        nb_steps = len(samples) // batch_size
        steps = 0

        for batch in generate_rows(computed_params, samples, batch_size, 2):
            X = batch[0]
            y = batch[1]['output']
            y = np.argmax(y, axis=-1)
            batch_samples = batch[3]

            y_pred = model.predict(X)
            y_pred = np.argmax(y_pred, axis=-1)

            for isample, sample in enumerate(batch_samples):
                sample.write(wrt)
                mark = u'' if y[isample]==y_pred[isample] else u' <-- ERROR'

                wrt.write(u'true_len={} predicted_len={} {}\n\n'.format(y[isample]+1, y_pred[isample]+1, mark))

            steps += 1
            if steps == nb_steps:
                break


PAD_WORD = u''


class GridGenerator(object):
    def __init__(self):
        pass

    def generate(self):
        params = dict()
        for optimizer in ['nadam']:
            params['optimizer'] = optimizer

            for batch_size in [200]:
                params['batch_size'] = batch_size

                for units1 in [100, 80, 64]:
                    params['units1'] = units1

                    for units2 in [80, 64, 0] if units1 > 0 else [0]:
                        params['units2'] = units2

                        for units3 in [16, 0] if units2 > 0 else [0]:
                            params['units3'] = units3

                            for net_arch in ['lstm']:  #, 'lstm(cnn)']:
                                params['net_arch'] = net_arch

                                if net_arch == 'lstm':
                                    for rnn_size in [200, 100]:
                                        params['rnn_size'] = rnn_size

                                        yield params
                                elif net_arch == 'lstm(cnn)':
                                    for rnn_size in [32, 64]:
                                        params['rnn_size'] = rnn_size

                                        for nb_filters in [100, 150]:
                                            params['nb_filters'] = nb_filters
                                            yield params
                                else:
                                    raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural model for answer length computation')
    parser.add_argument('--run_mode', type=str, default='train', choices='train query gridsearch'.split(), help='what to do: train | query | gridsearch')
    parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture: lstm lstm(cnn)')
    parser.add_argument('--batch_size', type=int, default=250, help='batch size for neural model training')
    parser.add_argument('--input', type=str, default='../data/pqa_all.dat', help='path to input dataset')
    parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
    parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
    parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')
    parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

    args = parser.parse_args()

    data_folder = args.data_dir
    input_path = args.input
    tmp_folder = args.tmp

    # настраиваем логирование в файл
    utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_answer_length.log'))

    wordchar2vector_path = args.wordchar2vector
    word2vector_path = os.path.expanduser(args.word2vector)
    batch_size = args.batch_size
    net_arch = args.arch
    run_mode = args.run_mode

    # В этих файлах будем сохранять натренированную сетку
    arch_filepath = os.path.join(tmp_folder, 'nn_answer_length.arch')
    weights_path = os.path.join(tmp_folder, 'nn_answer_length.weights')
    config_path = os.path.join(tmp_folder, 'nn_answer_length.config')

    if run_mode == 'train':
        logging.info('Start training...')

        # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
        samples, computed_params = load_samples(input_path)

        load_embeddings(tmp_folder, word2vector_path, computed_params)

        # сохраним конфиг модели, чтобы ее использовать в чат-боте
        model_config = {
                        'engine': 'nn',
                        'max_inputseq_len': computed_params['max_inputseq_len'],
                        'max_outputseq_len': computed_params['max_outputseq_len'],
                        'max_nb_premises': computed_params['max_nb_premises'],
                        'word2vector_path': word2vector_path,
                        'wordchar2vector_path': wordchar2vector_path,
                        'PAD_WORD': PAD_WORD,
                        'model_folder': tmp_folder,
                        'word_dims': computed_params['word_dims']
                       }

        with open(config_path, 'w') as f:
            json.dump(model_config, f)

        params = dict()
        params['net_arch'] = 'lstm'
        params['nb_filters'] = 128
        params['rnn_size'] = 150
        params['units1'] = 100
        params['units2'] = 0  #64
        params['units3'] = 0  #16
        params['optimizer'] = 'nadam'
        params['batch_size'] = 200

        model = create_model(params, computed_params)

        with open(arch_filepath, 'w') as f:
            f.write(model.to_json())

        keras.utils.plot_model(model,
                               to_file=os.path.join(tmp_folder, 'nn_answer_length.arch.png'),
                               show_shapes=False,
                               show_layer_names=True)

        SEED = 123456
        TEST_SHARE = 0.2
        train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)

        train_model(model, train_samples, val_samples, params, computed_params)

        val_acc = score_model(model, val_samples, params, computed_params)
        logging.info('Final val_acc={}'.format(val_acc))

        # Для контроля прогоним через модель все сэмплы и сохраним результаты в текстовый файл
        report_model(model, samples, params, os.path.join(tmp_folder, 'nn_answer_length.validation.txt'))

    if run_mode == 'gridsearch':
        logging.info('Start gridsearch')
        # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
        samples, computed_params = load_samples(input_path, 10000)
        load_embeddings(tmp_folder, word2vector_path, computed_params)

        best_params = None
        best_score = -np.inf
        crossval_count = 0

        grid = GridGenerator()
        for params in grid.generate():
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
                val_samples, finval_samples = train_test_split(val12_samples, test_size=0.5, random_state=SEED)

                model = create_model(params, computed_params)
                train_model(model, train_samples, val_samples, params, computed_params)

                score = score_model(model, finval_samples, params, computed_params)
                scores.append(score)

            score = np.mean(scores)
            score_std = np.std(scores)
            logging.info('Crossvalidation #{} score={} std={}'.format(crossval_count, score, score_std))
            if score > best_score:
                best_params = params.copy()
                best_score = score
                logging.info('!!! NEW BEST score={} params={}'.format(best_score, get_params_str(best_params)))

        logging.info(
            'Grid search complete, best_score={} best_params={}'.format(best_score, get_params_str(best_params)))

    if run_mode == 'query':
        logging.info('Start run_mode==query')

        with open(config_path, 'r') as f:
            model_config = json.load(f)
            max_inputseq_len = model_config['max_inputseq_len']
            w2v_path = model_config['word2vector_path']
            wordchar2vector_path = model_config['wordchar2vector_path']
            word_dims = model_config['word_dims']
            max_nb_premises = model_config['max_nb_premises']

        with open(arch_filepath, 'r') as f:
            model = model_from_json(f.read())

        model.load_weights(weights_path)

        computed_params = dict()
        load_embeddings(tmp_folder, w2v_path, computed_params)
        word_dims = computed_params['word_dims']
        word2vec = computed_params['word2vec']

        tokenizer = Tokenizer()
        tokenizer.load()

        Xn_probe = []
        for i in range(max_nb_premises + 1):
            Xn_probe.append(np.zeros((1, max_inputseq_len, word_dims), dtype='float32'))
        X_word = np.zeros((1, word_dims), dtype='float32')

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

                premises.append(premise)

            if question is None:
                question = utils.console_helpers.input_kbd('question:> ').strip().lower()
            if len(question) == 0:
                break

            question = question.replace(u'?', u'').strip()

            # Очистим входные тензоры перед заполнением новыми данными
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

            inputs = dict()
            for ipremise in range(max_nb_premises):
                inputs['premise{}'.format(ipremise)] = Xn_probe[ipremise]
            inputs['question'] = Xn_probe[max_nb_premises]
            inputs['word'] = X_word

            y_probe = model.predict(x=inputs)
            predicted_len = np.argmax(y_probe, axis=-1)
            predicted_len = predicted_len[0]+1  # длина с максимальной вероятностью
            p = y_probe[0][predicted_len-1]  # вероятность этой длины
            print('answer length={} (p={})'.format(predicted_len, p))

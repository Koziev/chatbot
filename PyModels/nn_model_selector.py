# -*- coding: utf-8 -*-
"""
Тренировка и ручная валидация модели классификатора, определяющего
способ генерации ответа в чат-боте (https://github.com/Koziev/chatbot).

Датасет должен быть предварительно сгенерирован скриптом prepare_qa_dataset.py
Также должны быть подготовлены векторы слов в посимвольной модели встраивания
с помощью vectorize_wordchar2vector.sh
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import sys

import itertools
import gensim
import keras.callbacks
import numpy as np
import tqdm
import argparse
import codecs
import logging


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.layers import Lambda
from keras.layers.merge import add, multiply

from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers


PAD_WORD = u''
padding = 'left'


class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer
        self.y = -1


def count_words(words):
    return len(filter(lambda z: z != PAD_WORD, words))


def lpad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def generate_rows(nb_premises, samples, batch_size, mode):
    batch_index = 0
    batch_count = 0

    Xn_batch = []
    for _ in range(nb_premises+1):
        x = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_batch.append(x)

    inputs = {}
    for ipremise in range(nb_premises):
        inputs['premise{}'.format(ipremise)] = Xn_batch[ipremise]
    inputs['question'] = Xn_batch[nb_premises]

    y_batch = np.zeros((batch_size, output_dims), dtype=np.bool)

    weights = np.zeros((batch_size))
    weights.fill(1.0)

    while True:
        for irow, sample in enumerate(samples):
            for ipremise, premise in enumerate(sample.premises):
                words = tokenizer.tokenize(premise)
                words = lpad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[ipremise], batch_index, word2vec)

            words = tokenizer.tokenize(sample.question)
            words = lpad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[nb_premises], batch_index, word2vec)

            y_batch[batch_index, sample.y] = True

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

# --------------------------------------------------------------------------------------


run_mode = ''
TRAIN_MODEL = 'model_selector'

parser = argparse.ArgumentParser(description='Neural model for answer generation model selector')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture: lstm | lstm(cnn)')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/pqa_all.dat', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.model', help='path to word2vector model file')

args = parser.parse_args()
input_path = args.input
tmp_folder = args.tmp
run_mode = args.run_mode
batch_size = args.batch_size
net_arch = args.arch
wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_model_selector.log'))

# В этих файлах будем сохранять натренированную сетку
arch_filepath = os.path.join(tmp_folder, 'qa_model_selector.arch')
weights_path = os.path.join(tmp_folder, 'qa_model_selector.weights')
config_path = os.path.join(tmp_folder, 'qa_model_selector.config')

# --------------------------------------------------------------------------

tokenizer = Tokenizer()
tokenizer.load()

if run_mode == 'train':
    logging.info('Start run_mode==train')

    logging.info(u'Loading samples from {}'.format(input_path))
    samples = []
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    max_inputseq_len = 0
    all_words = set()

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

                    answer_words = tokenizer.tokenize(answer)
                    sample.y = 2  # по умолчанию предполагается посимвольная генерация ответа
                    if len(answer_words) == 1 and answer_words[0] in [u'да', u'нет']:
                        sample.y = 0
                    elif all((c in u'0123456789') for c in answer):
                        sample.y = 3  # ответ - число, записанное цифрами.
                    else:
                        if len(premises) == 1:
                            premise_words = tokenizer.tokenize(sample.premises[0])
                            all_words_found = True
                            for answer_word in answer_words:
                                if answer_word not in premise_words:
                                    all_words_found = False
                                    break

                            if all_words_found:
                                sample.y = 1  # ответ копируется из единственной предпосылки

                    samples.append(sample)

                    max_nb_premises = max(max_nb_premises, len(premises))

                    for phrase in lines:
                        words = tokenizer.tokenize(phrase)
                        all_words.update(words)
                        max_inputseq_len = max(max_inputseq_len, len(words))

                    lines = []

            else:
                lines.append(line)

    logging.info('samples.count={}'.format(len(samples)))
    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_nb_premises={}'.format(max_nb_premises))

    logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    logging.info('wc2v_dims={0}'.format(wc2v_dims))

    for word in wc2v.vocab:
        all_words.add(word)

    nb_words = len(all_words)
    logging.info('nb_words={}'.format(nb_words))

    logging.info(u'Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    logging.info('w2v_dims={0}'.format(w2v_dims))

    word_dims = w2v_dims+wc2v_dims

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

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'engine': 'nn',
        'max_inputseq_len': max_inputseq_len,
        'max_nb_premises': max_nb_premises,
        'w2v_path': word2vector_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'model_folder': tmp_folder,
        'padding': padding,
        'word_dims': word_dims
    }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

        logging.info('Building the NN computational graph for {}...'.format(net_arch))

    nb_filters = 128
    rnn_size = word_dims

    inputs = []
    input_question = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='question')
    inputs.append(input_question)

    for ipremise in range(max_nb_premises):
        input_premise = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='premise{}'.format(ipremise))
        inputs.append(input_premise)

    layers = []
    encoder_size = 0

    if net_arch == 'lstm':
        # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения. Этот слой общий для всех входных предложений.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        for input in inputs:
            encoder_rnn = shared_words_rnn(input)
            layers.append(encoder_rnn)
            encoder_size += rnn_size*2

    # --------------------------------------------------------------------------

    if net_arch == 'lstm(cnn)':
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
                encoder_size += rnn_size

    classifier = keras.layers.concatenate(inputs=list(layers))

    # --------------------------------------------------------------------------

    # финальный классификатор определяет способ получения ответа:
    # 1) да/нет
    # 2) ответ строится копированием слов вопроса
    # 3) текст ответа генерируется сеткой
    # 4) ответ посимвольно генерируется сеткой и содержит одни цифры
    output_dims = 4

    classifier = Dense(encoder_size, activation='sigmoid')(classifier)
    #classifier = Dense(encoder_size//2, activation='relu')(classifier)
    #classifier = Dense(encoder_size//3, activation='relu')(classifier)
    classifier = Dense(output_dims, activation='softmax', name='output')(classifier)

    model = Model(inputs=inputs, outputs=classifier)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.summary()

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    # -------------------------------------------------------------------------

    SEED = 123456
    TEST_SHARE = 0.2
    train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)

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

    hist = model.fit_generator(generator=generate_rows(max_nb_premises, train_samples, batch_size, 1),
                               steps_per_epoch=nb_train_patterns // batch_size,
                               epochs=200,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(max_nb_premises, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns // batch_size
                               )
    max_acc = max(hist.history['val_acc'])
    logging.info('max val_acc={}'.format(max_acc))


if run_mode == 'query':
    # Ручное тестирование предварительно натренированной модели.
    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    with open(config_path, 'r') as f:
        model_config = json.load(f)

    max_inputseq_len = model_config['max_inputseq_len']
    word2vector_path = model_config['w2v_path']
    padding = model_config['padding']
    wordchar2vector_path = model_config['wordchar2vector_path']
    word_dims = model_config['word_dims']
    max_nb_premises = model_config['max_nb_premises']

    print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    print('wc2v_dims={0}'.format(wc2v_dims))

    print( 'Loading the w2v model {}'.format(word2vector_path) )
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))

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

    X1_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)

    while True:
        print('\nEnter two phrases:')
        phrase1 = raw_input('premise :> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase1) == 0:
            break

        phrase2 = raw_input('question:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase2) == 0:
            break

        words1 = tokenizer.tokenize(phrase1)
        words2 = tokenizer.tokenize(phrase2)

        if padding == 'left':
            words1 = lpad_wordseq(words1, max_inputseq_len)
            words2 = lpad_wordseq(words2, max_inputseq_len)
        else:
            words1 = rpad_wordseq(words1, max_inputseq_len)
            words2 = rpad_wordseq(words2, max_inputseq_len)

        X1_probe.fill(0)
        X2_probe.fill(0)

        vectorize_words(words1, X1_probe, 0, word2vec)
        vectorize_words(words2, X2_probe, 0, word2vec)
        y_probe = model.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})

        for i in range(3):
            print('y[{}]={}'.format(i, y_probe[0][i]))

        print('selected model={}'.format(np.argmax(y_probe[0])))

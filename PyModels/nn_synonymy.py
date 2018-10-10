# -*- coding: utf-8 -*-
"""
Модель для определения СИНОНИМИИ - семантической эквивалентности двух фраз,
включая позиционные и синтаксические перефразировки, лексические и фразовые
синонимы. В отличие от модели для РЕЛЕВАНТНОСТИ предпосылки и вопроса, в этой
модели предполагается, что объем информации в обеих фразах примерно одинаков,
то есть "кошка спит" и "черная кошка сладко спит" не считаются полными синонимами.

Для проекта чатбота https://github.com/Koziev/chatbot
"""

from __future__ import division
from __future__ import print_function

import codecs
import itertools
import json
import os
import sys
import argparse
import random
import collections
import logging

import numpy as np
import pandas as pd

import skimage.transform

import keras.callbacks
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.merge import concatenate, add, multiply
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
import keras.regularizers

from sklearn.model_selection import train_test_split
import sklearn.metrics

from utils.tokenizer import Tokenizer
from utils.padding_utils import lpad_wordseq, rpad_wordseq
from utils.padding_utils import PAD_WORD
from trainers.word_embeddings import WordEmbeddings
import utils.console_helpers
import utils.logging_helpers


use_shingle_matching = False

padding = 'left'

nb_neg_per_posit = 1

# размер изображения, которое получится после сжатия матрицы соответствия
# шинглов во входных предложениях.
shingle_image_size = 16


random.seed(123456789)
np.random.seed(123456789)


class Sample:
    def __init__(self, phrase1, phrase2, y):
        assert(len(phrase1) > 0)
        assert(len(phrase2) > 0)
        assert(y in [0, 1])
        self.phrase1 = phrase1
        self.phrase2 = phrase2
        self.y = y


def shingles_list(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def get_shingle_image(str1, str2):
    shingles1 = shingles_list(str1, 3)
    shingles2 = shingles_list(str2, 3)

    if len(shingles1) == 0 or len(shingles2) == 0:
        print('ERROR: empty string in get_shingle_image:')
        print(u'str1=', str1)
        print(u'str2=', str2)
        exit(1)

    image = np.zeros((len(shingles1), len(shingles2)), dtype='float32')
    for i1, shingle1 in enumerate(shingles1):
        for i2, shingle2 in enumerate(shingles2):
            if shingle1 == shingle2:
                image[i1, i2] = 1.0

    image_resized = skimage.transform.resize(image,
                                             (shingle_image_size, shingle_image_size))
    return image_resized.reshape(image_resized.size)


def ngrams(s, n):
    return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1&shingles2))/float(1e-8+len(shingles1|shingles2))


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


def vectorize_words(words, X_batch, irow, word2vec):
    for iword, word in enumerate(words):
        if word != PAD_WORD:
            X_batch[irow, iword, :] = word2vec[word]


def generate_rows(samples, batch_size, w2v, mode):
    if mode == 1:
        # При обучении сетки каждую эпоху тасуем сэмплы.
        random.shuffle(samples)

    batch_index = 0
    batch_count = 0

    X1_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    use_addfeatures = False
    if net_arch == 'cnn2':
        use_addfeatures = True
        X3_batch = np.zeros((batch_size, nb_addfeatures), dtype=np.float32)

    while True:
        for irow, sample in enumerate(samples):
            vectorize_words(sample.words1, X1_batch, batch_index, w2v)
            vectorize_words(sample.words2, X2_batch, batch_index, w2v)
            if mode == 1:
                y_batch[batch_index, sample.y] = True

            if use_addfeatures:
                iaddfeature = 0
                for word1 in sample.words1:
                    for word2 in sample.words2:
                        jaccard_sim = jaccard(word1, word2, 3)
                        X3_batch[batch_index, iaddfeature] = jaccard_sim
                        iaddfeature += 1

                        if word1 != PAD_WORD and word2 != PAD_WORD:
                            v1 = w2v[word1]
                            v2 = w2v[word2]
                            w2v_sim = v_cosine(v1, v2)
                            X3_batch[batch_index, iaddfeature] = w2v_sim
                        iaddfeature += 1

                if use_shingle_matching:
                    # добавляем сжатую матрицу соответствия шинглов
                    shingle_features = get_shingle_image(u' '.join(sample.words1), u' '.join(sample.words2))
                    iaddfeature2 = iaddfeature + shingle_features.shape[0]
                    X3_batch[batch_index, iaddfeature:iaddfeature2] = shingle_features
                    iaddfeature = iaddfeature2

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                xx = {'input_words1': X1_batch, 'input_words2': X2_batch}
                if use_addfeatures:
                    xx['input_addfeatures'] = X3_batch
                #print('DEBUG @704 yield batch_count={}'.format(batch_count))
                if mode == 1:
                    yield (xx, {'output': y_batch})
                else:
                    yield xx

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                if use_addfeatures:
                    X3_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0

# -------------------------------------------------------------------


NB_EPOCHS = 1000

# Разбор параметров тренировки в командной строке
parser = argparse.ArgumentParser(description='Neural model for short text synonymy')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query | query2')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture: lstm | lstm(cnn) | cnn | cnn2')
parser.add_argument('--classifier', type=str, default='merge', help='final classifier architecture: merge | muladd | merge2')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/synonymy_dataset.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.bin', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

args = parser.parse_args()
data_folder = args.data_dir
input_path = args.input
tmp_folder = args.tmp

wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)
batch_size = args.batch_size
net_arch = args.arch
classifier_arch = args.classifier

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_synonymy.log'))

run_mode = args.run_mode
# Варианты значения для параметра run_mode:
# run_mode='train'
# Тренировка модели по заранее приготовленному датасету с парами предложений
# run_mode = 'query'
# В консоли вводятся два предложения, модель оценивает их релевантность.


config_path = os.path.join(tmp_folder, 'nn_synonymy.config')

if run_mode == 'train':
    arch_filepath = os.path.join(tmp_folder, 'nn_synonymy.arch')
    weights_path = os.path.join(tmp_folder, 'nn_synonymy.weights')
else:
    # для не-тренировочных режимов загружаем ранее натренированную сетку
    # и остальные параметры модели.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_wordseq_len = int(model_config['max_wordseq_len'])
        word2vector_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        arch_filepath = model_config['arch_filepath']
        weights_path = model_config['weights_path']
        word_dims = model_config['word_dims']
        net_arch = model_config['net_arch']
        nb_addfeatures = model_config['nb_addfeatures']
        shingle_image_size = model_config['shingle_image_size']

    print('Restoring model architecture from {}'.format(arch_filepath))
    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    print('Loading model weights from {}'.format(weights_path))
    model.load_weights(weights_path)

# ------------------------------------------------------------------

tokenizer = Tokenizer()

if run_mode == 'train':
    logging.info('Start with run_mode==train')

    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
    word_dims = embeddings.vector_size

    # Грузим ранее подготовленный датасет для тренировки модели (см. prepare_synonymy_dataset.py)
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

    samples = []  # список из экземпляров Sample

    for phrase1, phrase2, y in itertools.izip(df['premise'].values, df['question'].values, df['relevance'].values):
        samples.append(Sample(phrase1, phrase2, y))

    nb_0 = sum(sample.y == 0 for sample in samples)
    nb_1 = sum(sample.y == 1 for sample in samples)
    logging.info('nb_0={} nb_1={}'.format(nb_0, nb_1))

    max_wordseq_len = 0
    for sample in samples:
        for phrase in [sample.phrase1, sample.phrase2]:
            words = tokenizer.tokenize(phrase)
            max_wordseq_len = max(max_wordseq_len, len(words))

    logging.info('max_wordseq_len={}'.format(max_wordseq_len))

    if padding == 'left':
        for sample in samples:
            sample.words1 = lpad_wordseq(tokenizer.tokenize(sample.phrase1), max_wordseq_len)
            sample.words2 = lpad_wordseq(tokenizer.tokenize(sample.phrase2), max_wordseq_len)
    else:
        for sample in samples:
            sample.words1 = rpad_wordseq(tokenizer.tokenize(sample.phrase1), max_wordseq_len)
            sample.words2 = rpad_wordseq(tokenizer.tokenize(sample.phrase2), max_wordseq_len)


    # суммарное кол-во дополнительных фич, подаваемых на вход сетки
    # помимо двух отдельных предложений.
    nb_addfeatures = 0

    if net_arch == 'cnn2':
        # попарные похожести слов в двух предложениях.
        for i1 in range(max_wordseq_len):
            for i2 in range(max_wordseq_len):
                nb_addfeatures += 1  # жаккардова похожесть
                nb_addfeatures += 1  # w2v cosine

        if use_shingle_matching:
            # визуальное представление паттернов Жаккара по шинглам.
            nb_addfeatures += shingle_image_size*shingle_image_size

        logging.info('nb_addfeatures={}'.format(nb_addfeatures))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'model': 'nn',
        'max_wordseq_len': max_wordseq_len,
        'w2v_path': word2vector_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'padding': padding,
        'arch_filepath': arch_filepath,
        'weights_path': weights_path,
        'word_dims': word_dims,
        'net_arch': net_arch,
        'nb_addfeatures': nb_addfeatures,
        'shingle_image_size': shingle_image_size,
        'use_shingle_matching': use_shingle_matching
    }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    logging.info('Constructing the NN model arch={} classifier={}...'.format(net_arch, classifier_arch))

    nb_filters = 128  # 128
    rnn_size = word_dims*2

    classif = None
    sent2vec_input = None
    sent2vec_output = None

    words_net1 = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_words1')
    words_net2 = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_words2')
    addfeatures_input = None
    if net_arch == 'cnn2':
        addfeatures_input = Input(shape=(nb_addfeatures,), dtype='float32', name='input_addfeatures')

    sent2vec_input = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input')

    # группы слоев с для первого и второго предложения соответственно
    conv1 = []
    conv2 = []

    # суммарный размер выходных тензоров в conv1, то есть это сумма размеров векторов
    # для всех слоев в списке conv1, если их смерджить.
    encoder_size = 0

    # группа слоев для модели векторизации предложения.
    sent2vec_conv = []

    if net_arch == 'ff':
        # feedforward сетка.
        # входную последовательность векторов слов в каждом предложении
        # развернем в вектор, который подаем на несколько простых полносвязных
        # слоев.
        encoder1 = Flatten()(words_net1)
        encoder2 = Flatten()(words_net2)

        sent2vec_encoder = Flatten()(sent2vec_input)

        nb_inputs = max_wordseq_len*word_dims

        for i in range(1):
            shared_layer = Dense(units=nb_inputs, activation='relu', name='shared_dense[{}]'.format(i))

            #if i == 1:
            #    encoder1 = BatchNormalization()(encoder1)
            #    encoder2 = BatchNormalization()(encoder2)

            encoder1 = shared_layer(encoder1)
            encoder2 = shared_layer(encoder2)
            sent2vec_encoder = shared_layer(sent2vec_encoder)

        encoder_size = nb_inputs
        conv1.append(encoder1)
        conv2.append(encoder2)
        sent2vec_conv.append(sent2vec_encoder)

    if net_arch == 'lstm':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_wordseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        encoder_size = rnn_size
        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)

        sent2vec_output = shared_words_rnn(sent2vec_input)
        sent2vec_conv.append(sent2vec_output)

    if net_arch == 'lstm(lstm)':
        # два рекуррентных слоя
        shared_words_rnn1 = Bidirectional(recurrent.LSTM(rnn_size,
                                                         input_shape=(max_wordseq_len, word_dims),
                                                         return_sequences=True))

        shared_words_rnn2 = Bidirectional(recurrent.LSTM(rnn_size,
                                                         return_sequences=False))


        encoder_rnn1 = shared_words_rnn1(words_net1)
        encoder_rnn1 = shared_words_rnn2(encoder_rnn1)

        encoder_rnn2 = shared_words_rnn1(words_net2)
        encoder_rnn2 = shared_words_rnn2(encoder_rnn2)

        encoder_size = rnn_size
        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)

        sent2vec_output = shared_words_rnn1(sent2vec_input)
        sent2vec_output = shared_words_rnn2(sent2vec_output)
        sent2vec_conv.append(sent2vec_output)

    if net_arch == 'cnn(lstm)':
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_wordseq_len, word_dims),
                                                        return_sequences=True))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        sent2vec_rnn2 = shared_words_rnn(sent2vec_input)

        for kernel_size in range(2, 5):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            #pooler = GlobalMaxPooling1D()
            pooler = GlobalAveragePooling1D()

            conv_layer1 = conv(encoder_rnn1)
            conv_layer1 = pooler(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(encoder_rnn2)
            conv_layer2 = pooler(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += nb_filters

            sent2vec_output = conv(sent2vec_rnn2)
            sent2vec_output = pooler(sent2vec_output)

            sent2vec_conv.append(sent2vec_output)

    if net_arch == 'cnn':
        # простая сверточная архитектура.
        for kernel_size in range(1, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            #pooler = GlobalMaxPooling1D()
            pooler = GlobalAveragePooling1D()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv2.append(conv_layer2)

            #sent2vec_encoder = conv(sent2vec_input)
            #sent2vec_encoder = GlobalMaxPooling1D()(sent2vec_input)
            #sent2vec_conv.append(sent2vec_encoder)

            encoder_size += nb_filters

        #print('DEBUG len(sent2vec_conv)={} encoder_size={}'.format(len(sent2vec_conv), encoder_size))

    if net_arch == 'cnn2':
        for kernel_size, nb_filters in [(1, 100), (2, 200), (3, 400), (4, 1000)]:
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            #pooler = GlobalMaxPooling1D()
            pooler = GlobalAveragePooling1D()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += nb_filters

    if net_arch == 'cnn(cnn)':
        # двухслойная сверточная сетка
        for kernel_size in range(1, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='relu',
                          strides=1)

            pooler = AveragePooling1D()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv2.append(conv_layer2)

        conv1_1 = keras.layers.concatenate(inputs=conv1)
        conv2_1 = keras.layers.concatenate(inputs=conv2)

        conv = Conv1D(filters=80,
                      kernel_size=2,
                      padding='valid',
                      activation='relu',
                      strides=1)

        pooler = GlobalAveragePooling1D()

        conv_layer1 = conv(conv1_1)
        conv_layer1 = pooler(conv_layer1)
        conv1 = []
        conv1.append(conv_layer1)

        conv_layer2 = conv(conv2_1)
        conv_layer2 = pooler(conv_layer2)
        conv2 = []
        conv2.append(conv_layer2)

        encoder_size += 80

        #print('DEBUG len(sent2vec_conv)={} encoder_size={}'.format(len(sent2vec_conv), encoder_size))

    if net_arch == 'lstm+cnn':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_wordseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        #dense1 = Dense(units=rnn_size*2)
        #encoder_rnn1 = dense1(encoder_rnn1)
        #encoder_rnn2 = dense1(encoder_rnn2)

        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)

        encoder_size += rnn_size*2

        # добавляем входы со сверточными слоями
        for kernel_size in range(2, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            #dense2 = Dense(units=nb_filters)

            conv_layer1 = conv(words_net1)
            conv_layer1 = GlobalMaxPooling1D()(conv_layer1)
            #conv_layer1 = dense2(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = GlobalMaxPooling1D()(conv_layer2)
            #conv_layer2 = dense2(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += nb_filters

    if net_arch == 'lstm(cnn)':

        if False:
            shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                            input_shape=(max_wordseq_len, word_dims),
                                                            return_sequences=False))

            encoder_rnn1 = shared_words_rnn(words_net1)
            encoder_rnn2 = shared_words_rnn(words_net2)

            conv1.append(encoder_rnn1)
            conv2.append(encoder_rnn2)
            encoder_size += rnn_size*2

            sent2vec_layer = shared_words_rnn(sent2vec_input)
            sent2vec_conv.append(sent2vec_layer)

        for kernel_size in range(1, 4):
            # сверточные слои, образующие детекторы словосочетаний
            # и синтаксических конструкций.
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            pooler = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')
            #pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')
            #pooler = None

            # поверх сверточных идут рекуррентные слои
            lstm = recurrent.LSTM(rnn_size, return_sequences=False)

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv_layer2 = lstm(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += rnn_size

            sent2vec_layer = conv(sent2vec_input)
            sent2vec_layer = pooler(sent2vec_layer)
            sent2vec_layer = lstm(sent2vec_layer)
            sent2vec_conv.append(sent2vec_layer)

        logging.info('encoder_size={}'.format(encoder_size))

    # --------------------------------------------------------------------------
    # Далее идут разные варианты завершающей части сетки для определения релевантности,
    # а именно - классификатор релевантности. У него на входе два набора тензоров
    # в conv1 и conv2, выход - бинарная классификация.
    # if'ами переключаются разные архитектуры этой части.

    sent2vec_dim = min(encoder_size, 128)  # такая длина будет у вектора представления предложения

    activity_regularizer = None  #keras.regularizers.l1(0.000001)
    sent_repr_layer = Dense(units=sent2vec_dim,
                            activation='relu',
                            activity_regularizer=activity_regularizer,
                            name='sentence_representation')

    if classifier_arch == 'merge2':
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
        encoder1 = sent_repr_layer(encoder1)
        encoder2 = sent_repr_layer(encoder2)

        addition = add([encoder1, encoder2])
        minus_y1 = Lambda(lambda x: -x, output_shape=(sent2vec_dim,))(encoder1)
        mul = add([encoder2, minus_y1])
        mul = multiply([mul, mul])

        #words_final = keras.layers.concatenate(inputs=[mul, addition, addfeatures_input])
        words_final = keras.layers.concatenate(inputs=[mul, addition, addfeatures_input, encoder1, encoder2])
        final_size = encoder_size+nb_addfeatures
        words_final = Dense(units=final_size//2, activation='sigmoid')(words_final)

    elif classifier_arch == 'merge':
        # этот финальный классификатор берет два вектора представления предложений,
        # объединяет их в вектор двойной длины и затем прогоняет этот двойной вектор
        # через несколько слоев.
        if net_arch == 'cnn2':
            #addfeatures_layer = Dense(units=nb_addfeatures, activation='relu')(addfeatures_input)
            encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2, [addfeatures_input])))
        else:
            if len(conv1) == 1:
                encoder_merged = keras.layers.concatenate(inputs=[conv1[0], conv2[0]])
            else:
                encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2)))

        final_size = encoder_size*2+nb_addfeatures
        #words_final = Dense(units=final_size, activation='sigmoid')(encoder_merged)
        #words_final = Dense(units=final_size, activation='relu')(encoder_merged)
        words_final = encoder_merged
        # words_final = BatchNormalization()(words_final)
        words_final = Dense(units=final_size//2, activation='relu')(words_final)
        #words_final = BatchNormalization()(words_final)
        words_final = Dense(units=encoder_size//3, activation='relu')(words_final)
        #words_final = BatchNormalization()(words_final)
        #words_final = Dense(units=encoder_size//3, activation='relu')(words_final)

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
        encoder1 = sent_repr_layer(encoder1)
        encoder2 = sent_repr_layer(encoder2)

        addition = add([encoder1, encoder2])
        minus_y1 = Lambda(lambda x: -x, output_shape=(sent2vec_dim,))(encoder1)
        mul = add([encoder2, minus_y1])
        mul = multiply([mul, mul])

        #words_final = keras.layers.concatenate(inputs=[encoder1, mul, addition, encoder2])
        words_final = keras.layers.concatenate(inputs=[mul, addition])  # эксперимент!!!
        words_final = Dense(units=sent2vec_dim, activation='relu')(words_final)
        words_final = Dense(units=sent2vec_dim // 2, activation='relu')(words_final)
        words_final = Dense(units=sent2vec_dim // 3, activation='relu')(words_final)
        #words_final = Dense(units=sent2vec_dim // 4, activation='relu')(words_final)
        #words_final = Dense(units=sent2vec_dim // 5, activation='relu')(words_final)
    else:
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

        encoder_merged1 = sent_repr_layer(encoder1)
        encoder_merged2 = sent_repr_layer(encoder2)

        if classifier_arch == 'mul':
            # вариант с почленным произведением двух векторов
            words_final = keras.layers.multiply(inputs=[encoder_merged1, encoder_merged2])
            words_final = Dense(units=sent2vec_dim//2, activation='relu')(words_final)
            words_final = Dense(units=sent2vec_dim//4, activation='relu')(words_final)
        elif classifier_arch == 'l2':
            # L2 норма разности между двумя векторами.
            words_final = keras.layers.Lambda(lambda v12: K.sqrt(K.sum((v12[0] - v12[1]) ** 2)))([encoder_merged1, encoder_merged2])
        elif classifier_arch == 'subtract':
            # вариант с разностью векторов
            words_final = keras.layers.subtract(inputs=[encoder_merged1, encoder_merged2])
            words_final = Dense(units=sent2vec_dim//2, activation='relu')(words_final)
            words_final = Dense(units=sent2vec_dim//4, activation='relu')(words_final)
        elif classifier_arch == 'l2_2':
            # другой вариант вычисления L2 для разности векторов
            words_final = keras.layers.subtract(inputs=[encoder_merged1, encoder_merged2])
            classif = keras.layers.Lambda(lambda x: K.sqrt(K.sum(x ** 2, axis=1)))(words_final)
            #classif = keras.layers.Lambda(lambda x: K.exp(-x))(words_final)
        elif classifier_arch == 'dot':
            # скалярное произведение двух векторов, дающее скаляр.
            words_final = keras.layers.dot(inputs=[encoder_merged1, encoder_merged2], axes=1, normalize=True)

    # Вычислительный граф сформирован, добавляем финальный классификатор с 1 выходом,
    # выдающим 0 для нерелевантных и 1 для релевантных предложений.
    if classif is None:
        #classif = Dense(units=1, activation='sigmoid', name='output')(words_final)
        classif = Dense(units=2, activation='softmax', name='output')(words_final)

    if addfeatures_input is None:
        xx = [words_net1, words_net2]
    else:
        xx = [words_net1, words_net2, addfeatures_input]
    model = Model(inputs=xx, outputs=classif)

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    SEED = 123456
    TEST_SHARE = 0.3
    train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)
    val_samples, eval_samples = train_test_split(val_samples, test_size=0.3, random_state=SEED)
    logging.info('train_samples.count={}'.format(len(train_samples)))
    logging.info('val_samples.count={}'.format(len(val_samples)))
    logging.info('eval_samples.count={}'.format(len(eval_samples)))

    logging.info('Start training...')
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc',
                                   patience=10,
                                   verbose=1,
                                   mode='auto')

    nb_validation_steps = len(val_samples)//batch_size

    if True:
        hist = model.fit_generator(generator=generate_rows(train_samples, batch_size, embeddings, 1),
                                   steps_per_epoch=len(train_samples)//batch_size,
                                   epochs=NB_EPOCHS,
                                   verbose=1,
                                   callbacks=[model_checkpoint, early_stopping],
                                   validation_data=generate_rows(val_samples, batch_size, embeddings, 1),
                                   validation_steps=nb_validation_steps,
                                   )
        max_acc = max(hist.history['val_acc'])
        logging.info('max val_acc={}'.format(max_acc))

        # загрузим чекпоинт с оптимальными весами
        model.load_weights(weights_path)

    logging.info('Estimating final f1 score...')
    # получим оценку F1 на валидационных данных
    y_true2 = []
    y_pred2 = []
    for istep, xy in enumerate(generate_rows(val_samples, batch_size, embeddings, 1)):
        x = xy[0]
        y = xy[1]['output']
        y_pred = model.predict(x=x, verbose=0)
        for k in range(len(y_pred)):
            y_true2.append(y[k][1])
            y_pred2.append(y_pred[k][1] > y_pred[k][0])

            #irow = len(y_true2)
            #if irow == 195:  # <-- DEBUG
            #    wrt.write('DEBUG:\nx1={}\nx2={}\n\n'.format(x['input_words1'][irow], x['input_words2'][irow]))

        if istep >= nb_validation_steps:
            break

    with codecs.open(os.path.join(tmp_folder, 'nn_synonymy.validation.txt'), 'w', 'utf-8') as wrt:
        for irow in range(len(val_samples)):
            wrt.write(u'isample={}\n'.format(irow))
            wrt.write(u'{}\n'.format(val_samples[irow].phrase1))
            wrt.write(u'{}\n'.format(val_samples[irow].phrase2))
            wrt.write(u'y_true={} y_model={}\n\n'.format(y_true2[irow], y_pred2[irow]))

    # из-за сильного дисбаланса (в пользу исходов с y=0) оценивать качество
    # получающейся модели лучше по f1
    f1 = sklearn.metrics.f1_score(y_true=y_true2, y_pred=y_pred2)
    logging.info('val f1={}'.format(f1))

# </editor-fold>

# <editor-fold desc="query">
if run_mode == 'query':
    ################################################################
    # Ввод двух предложений с клавиатуры и выдача их синонимичности.
    ################################################################

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_inputseq_len = model_config['max_wordseq_len']
        w2v_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        net_arch = model_config['net_arch']
        nb_addfeatures = model_config['nb_addfeatures']
        shingle_image_size = model_config['shingle_image_size']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
    word_dims = embeddings.vector_size

    while True:
        print('\nEnter two phrases:')
        phrase1 = raw_input('phrase #1:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase1) == 0:
            break

        phrase2 = raw_input('phrase #2:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase2) == 0:
            break

        sample = Sample(phrase1, phrase2, 0)
        if padding == 'left':
            sample.words1 = lpad_wordseq(tokenizer.tokenize(phrase1), max_wordseq_len)
            sample.words2 = lpad_wordseq(tokenizer.tokenize(phrase2), max_wordseq_len)
        else:
            sample.words1 = rpad_wordseq(tokenizer.tokenize(phrase1), max_wordseq_len)
            sample.words2 = rpad_wordseq(tokenizer.tokenize(phrase2), max_wordseq_len)

        samples = [sample]
        for data in generate_rows(samples, 1, embeddings, 1):
            x_probe = data[0]

            # <-- DEBUG
            #print('DEBUG:\nx1={}\nx2={}\n\n'.format(x_probe['input_words1'], x_probe['input_words2']))
            #<-- DEBUG END

            y_probe = model.predict(x=x_probe)
            sim = y_probe[0]  # Получится вектор из 2 чисел: p(несинонимичны) p(синонимичны)
            print('sim={}'.format(sim))
            break

# </editor-fold>


# <editor-fold desc="query2">
if run_mode == 'query2':
    # В консоли вводится предложение, для которого в списке smalltalk.txt
    # ищутся ближайшие.

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_inputseq_len = model_config['max_wordseq_len']
        w2v_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        net_arch = model_config['net_arch']
        nb_addfeatures = model_config['nb_addfeatures']
        shingle_image_size = model_config['shingle_image_size']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
    word_dims = embeddings.vector_size

    phrases2 = []
    with codecs.open(os.path.join(data_folder, 'smalltalk.txt'), 'r', 'utf-8') as rdr:
        for line in rdr:
            phrase = line.strip()
            if len(phrase) > 5 and phrase.startswith(u'Q:'):
                phrase = u' '.join(tokenizer.tokenize(phrase.replace(u'Q:', u'')))
                phrases2.append(phrase)

    while True:
        phrase1 = raw_input('phrase:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase1) == 0:
            break

        samples = []
        premise_samples = []
        question_samples = []
        Sample2 = collections.namedtuple('Sample2', ['words1', 'words2'])
        for phrase2 in phrases2:
            if padding == 'left':
                words1 = lpad_wordseq(tokenizer.tokenize(phrase1), max_wordseq_len)
                words2 = lpad_wordseq(tokenizer.tokenize(phrase2), max_wordseq_len)
            else:
                words1 = rpad_wordseq(tokenizer.tokenize(phrase1), max_wordseq_len)
                words2 = rpad_wordseq(tokenizer.tokenize(phrase2), max_wordseq_len)

            samples.append(Sample2(words1, words2))

        nb_samples = len(samples)

        print('Vectorization of {} samples'.format(nb_samples))
        for data in generate_rows(samples, nb_samples, embeddings, 2):
            print('Running model to compute similarity of premises')
            sims = model.predict(x=data, batch_size=batch_size)[:, 1]
            break

        phrase_sims = [(phrases2[i], sims[i]) for i in range(len(sims))]
        phrase_sims = sorted(phrase_sims, key=lambda z: -z[1])
        for phrase, sim in phrase_sims[:10]:
            print(u'{:6.4f} {}'.format(sim, phrase))

# </editor-fold>

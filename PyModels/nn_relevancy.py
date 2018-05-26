# -*- coding: utf-8 -*-
"""
Модель для определения релевантности предпосылки и вопроса.
Модель используется в проекте чат-бота https://github.com/Koziev/chatbot

Вторая функция программы - обучение модели для генерации вектора предложения (sent2vec).

Датасет должен быть предварительно сгенерирован скриптом prepare_relevancy_dataset.py

Слова представляются векторами word2vector и wordchar2vector моделей. Списки соответствующих
векторов читаются из текстовых файлов. Обучение wordchar2vector модели и генерация векторов
для используемого списка слов должны быть выполнены заранее скриптом wordchar2vector.py в
корне проекте.
"""

from __future__ import division
from __future__ import print_function

import codecs
import gc
import itertools
import json
import os
import sys
import argparse

import gensim
import numpy as np
import pandas as pd
import tqdm

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

from utils.tokenizer import Tokenizer
from utils.segmenter import Segmenter
from utils.padding_utils import pad_wordseq
from utils.padding_utils import PAD_WORD
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup


config_file_name = 'nn_relevancy_model.config'
config_file_name2 = 'sent2vec.config'


# размер изображения, которое получится после сжатия матрицы соответствия
# шинглов во входных предложениях.
shingle_image_size = 16


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


def load_word_vectors(wordchar2vector_path, word2vector_path):
    # Грузим заранее подготовленные векторы слов для модели
    # встраивания wordchar2vector (см. wordchar2vector.py)
    print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    print('wc2v_dims={0}'.format(wc2v_dims))

    # --------------------------------------------------------------------------

    print( 'Loading the w2v model {}'.format(word2vector_path) )
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))

    # Векторы слов получаются соединением векторов моделей word2vector и wordchar2vector
    word_dims = w2v_dims+wc2v_dims

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros( word_dims )
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    del wc2v
    gc.collect()

    return (word2vec, word_dims, w2v)

# -------------------------------------------------------------------

# Разбор параметров тренировки, указанных в командной строке
parser = argparse.ArgumentParser(description='Neural model for text relevance estimation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train evaluate query query2')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture, one of lstm lstm(cnn) (lstm)cnn cnn')
parser.add_argument('--classifier', type=str, default='merge', help='final classifier architecture')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--max_nb_samples', type=int, default=1000000000, help='upper limit for number of samples')
parser.add_argument('--input', type=str, default='../data/premise_question_relevancy.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='/home/eek/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.model', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')


args = parser.parse_args()
data_folder = args.data_dir
input_path = args.input
tmp_folder = args.tmp

wordchar2vector_path = args.wordchar2vector
word2vector_path = args.word2vector
batch_size = args.batch_size
net_arch = args.arch
classifier_arch = args.classifier

# Для быстрого проведения исследований влияния гиперпараметров удобно брать для
# обучения не полный датасет, а небольшую часть - указываем кол-во паттернов.
max_nb_samples = args.max_nb_samples

run_mode = args.run_mode
# Варианты значения для параметра run_mode:
# run_mode='train'
# Тренировка модели по заранее приготовленному датасету с парами предложений

# run_mode = 'evaluate'
# Оценка качества натренированной модели через анализ того, насколько хорошо модель
# выбирает ожидаемые предпосылки под тестовые вопросы против остальных предпосылок.

# run_mode = 'query'
# В консоли вводятся два предложения, модель оценивает их релевантность.

# run_mode = 'query2'
# В консоли вводится имя текстового файла со списком предложений
# и второе проверяемое предложение. Модель выводит список предложений,
# отсортированный по релевантности.


if run_mode == 'train':
    arch_filepath = os.path.join(tmp_folder, 'nn_relevancy_model.arch')
    weights_path = os.path.join(tmp_folder, 'nn_relevancy.weights')

    # 30.04.2018 отдельно сохраним модель для генерации вектора по тексту предложения.
    arch_filepath2 = os.path.join(tmp_folder, 'sent2vector.arch')
    weights_path2 = os.path.join(tmp_folder, 'sent2vector.weights')
else:
    # для не-тренировочных режимов загружаем ранее натренированную сетку
    # и остальные параметры модели.

    with open(os.path.join(tmp_folder, config_file_name), 'r') as f:
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

    word2vec, word_dims, w2v = load_word_vectors(wordchar2vector_path, word2vector_path)

    # Грузим ранее подготовленный датасет для тренировки модели
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

    # Анализ и векторизация датасета
    all_words = set()
    max_wordseq_len = 0
    for phrase in itertools.chain(df['premise'].values, df['question'].values):
        words = tokenizer.tokenize(phrase)
        all_words.update(words)
        max_wordseq_len = max(max_wordseq_len, len(words))

    for word in word2vec:
        all_words.add(word)

    print('max_wordseq_len={}'.format(max_wordseq_len))

    nb_words = len(all_words)
    print('nb_words={}'.format(nb_words))

    # суммарное кол-во дополнительных фич, подаваемых на вход сетки
    # помимо двух отдельных предложений.
    nb_addfeatures = 0

    if net_arch == 'cnn2':
        # попарные похожести слов в двух предложениях.
        for i1 in range(max_wordseq_len):
            for i2 in range(max_wordseq_len):
                nb_addfeatures += 1  # жаккардова похожесть
                nb_addfeatures += 1  # w2v cosine
        # визуальное представление паттернов Жаккара по шинглам.
        nb_addfeatures += shingle_image_size*shingle_image_size

        print('nb_addfeatures={}'.format(nb_addfeatures))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'model': 'nn',
        'max_wordseq_len': max_wordseq_len,
        'w2v_path': word2vector_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'arch_filepath': arch_filepath,
        'weights_path': weights_path,
        'word_dims': word_dims,
        'net_arch': net_arch,
        'nb_addfeatures': nb_addfeatures,
        'shingle_image_size': shingle_image_size
    }

    with open(os.path.join(tmp_folder, config_file_name), 'w') as f:
        json.dump(model_config, f)

    print('Constructing the NN model {} {}...'.format(net_arch, classifier_arch))

    nb_filters = 128  # 128
    rnn_size = word_dims

    classif = None
    sent2vec_input = None
    sent2vec_output = None

    words_net1 = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_words1')
    words_net2 = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_words2')

    if net_arch == 'cnn2':
        addfeatures_input = Input(shape=(nb_addfeatures,), dtype='float32', name='input_addfeatures')

    sent2vec_input = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input')

    # группы слоев с для первого и второго предложения соотственно
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

    if net_arch == '(lstm)cnn':
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

            conv_layer1 = conv(encoder_rnn1)
            conv_layer1 = GlobalMaxPooling1D()(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(encoder_rnn2)
            conv_layer2 = GlobalMaxPooling1D()(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += nb_filters

            sent2vec_output = conv(sent2vec_rnn2)
            sent2vec_output = GlobalMaxPooling1D()(sent2vec_output)

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

        print('encoder_size={}'.format(encoder_size))

    # --------------------------------------------------------------------------
    # Далее идут разные варианты завершающей части сетки для определения релевантности,
    # а именно - классификатор релевантности. У него на входе два набора тензоров
    # в conv1 и conv2, выход - бинарная классификация.
    # if'ами переключаются разные архитектуры этой части.

    sent2vec_dim = 128  # такая длина будет у вектора представления предложения

    activity_regularizer = None  #keras.regularizers.l1(0.000001)
    sent_repr_layer = Dense(units=sent2vec_dim,
                            activation='relu',
                            activity_regularizer=activity_regularizer,
                            name='sentence_representation')

    if classifier_arch == 'merge':
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

        words_final = Dense(units=encoder_size*2+nb_addfeatures, activation='relu')(encoder_merged)
        # words_final = BatchNormalization()(words_final)
        words_final = Dense(units=encoder_size, activation='relu')(words_final)
        # words_final = BatchNormalization()(words_final)
        words_final = Dense(units=encoder_size//2, activation='relu')(words_final)
        # words_final = BatchNormalization()(words_final)
        words_final = Dense(units=encoder_size//3, activation='relu')(words_final)

        if len(sent2vec_conv) > 1:
            if len(sent2vec_conv) == 1:
                sent2vec_output = sent2vec_conv[0]
            else:
                sent2vec_output = keras.layers.concatenate(inputs=sent2vec_conv)

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

        if len(sent2vec_conv) > 0:
            if len(sent2vec_conv) == 1:
                sent2vec_output = sent2vec_conv[0]
            else:
                sent2vec_output = keras.layers.concatenate(inputs=sent2vec_conv)

            sent2vec_output = sent_repr_layer(sent2vec_output)

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

        sent2vect_merged = keras.layers.concatenate(inputs=sent2vec_conv)
        sent2vec_output = sent_repr_layer(sent2vect_merged)

    # Вычислительный граф сформирован, добавляем финальный классификатор с 1 выходом,
    # выдающим 0 для нерелевантных и 1 для релевантных предложений.
    if classif is None:
        #classif = Dense(units=1, activation='sigmoid', name='output')(words_final)
        classif = Dense(units=2, activation='softmax', name='output')(words_final)

    model = Model(inputs=[words_net1, words_net2, addfeatures_input], outputs=classif)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

# -----------------------------------------------------------------


def vectorize_words(words, X_batch, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            X_batch[irow, iword, :] = word2vec[word]


def generate_rows(sequences, targets, batch_size, w2v, mode):
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
        for irow, (seq, target) in enumerate(itertools.izip(sequences, targets)):
            vectorize_words(seq[0], X1_batch, batch_index, word2vec)
            vectorize_words(seq[1], X2_batch, batch_index, word2vec)
            y_batch[batch_index, target] = True

            if use_addfeatures:
                iaddfeature = 0
                for word1 in seq[0]:
                    for word2 in seq[1]:
                        jaccard_sim = jaccard(word1, word2, 3)
                        X3_batch[batch_index, iaddfeature] = jaccard_sim
                        iaddfeature += 1

                        w2v_sim = 0.0
                        if word1 in w2v and word2 in w2v:
                            v1 = w2v[word1]
                            v2 = w2v[word2]
                            w2v_sim = v_cosine(v1, v2)
                        X3_batch[batch_index, iaddfeature] = w2v_sim
                        iaddfeature += 1

                # добавляем сжатую матрицу соответствия шинглов
                shingle_features = get_shingle_image( u' '.join(seq[0]), u' '.join(seq[1]) )
                iaddfeature2 = iaddfeature + shingle_features.shape[0]
                X3_batch[batch_index, iaddfeature:iaddfeature2] = shingle_features
                iaddfeature = iaddfeature2

            batch_index += 1

            #print('DEBUG @697 batch_index={} batch_size={}'.format(batch_index, batch_size))

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

    #print('DEBUG @717 generate_rows exhausted')
    #raise StopIteration()

# ---------------------------------------------------------------

# <editor-fold desc="train">
if run_mode == 'train':
    phrases = []
    ys = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
        phrase1 = row['premise']
        phrase2 = row['question']
        words1 = pad_wordseq(tokenizer.tokenize(phrase1), max_wordseq_len)
        words2 = pad_wordseq(tokenizer.tokenize(phrase2), max_wordseq_len)

        if len(phrase1) < 3 or len(phrase2) < 3:
            print(u'Empty word sequence in sample #{}:\nphrase1={}\nphrase2={}'.format(index, phrase1, phrase2))
            print(u'words1=', words1)
            print(u'words2=', words2)
            exit(2)

        y = row['relevance']
        if y in (0, 1):
            ys.append(y)
            phrases.append((words1, words2, phrase1, phrase2))

    if len(phrases) > max_nb_samples:
        print('Reducing the list of samples from {} to {} items'.format(len(phrases), max_nb_samples))
        # iphrases = list(np.random.permutation(range(len(phrases))))
        phrases = phrases[:max_nb_samples]
        ys = ys[:max_nb_samples]

    SEED = 123456
    TEST_SHARE = 0.2
    train_phrases, val_phrases, train_ys, val_ys = train_test_split(phrases,
                                                                    ys,
                                                                    test_size=TEST_SHARE,
                                                                    random_state=SEED)

    print('train_phrases.count={}'.format(len(train_phrases)))
    print('val_phrases.count={}'.format(len(val_phrases)))

    print('Start training...')
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_acc',
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')

    hist = model.fit_generator(generator=generate_rows(train_phrases, train_ys, batch_size, w2v, 1),
                               steps_per_epoch=int(len(train_phrases)/batch_size),
                               epochs=100,
                               verbose=1,
                               callbacks=[model_checkpoint, early_stopping],
                               validation_data=generate_rows( val_phrases, val_ys, batch_size, w2v, 1),
                               validation_steps=int(len(val_phrases)/batch_size),
                               )
    max_acc = max(hist.history['val_acc'])
    print('max val_acc={}'.format(max_acc))

    if sent2vec_output is not None:
        # загрузим чекпоинт с оптимальными весами, построим новую модель sent2vector и сохраним
        # ее на диск.
        model.load_weights(weights_path)

        # сохраняем модель sent2vec
        sent2vec_model = Model(inputs=sent2vec_input, outputs=sent2vec_output)
        sent2vec_model.compile(loss='mse', optimizer='nadam', metrics=['accuracy'])
        with open(arch_filepath2, 'w') as f:
            f.write(sent2vec_model.to_json())
        sent2vec_model.save_weights(weights_path2)

        #sent2vec_model.summary()

        # А давайте-ка посмотрим на статистику значений компонент в получающихся
        # векторах предложений.
        # Для этого прогоним через модель все фразы.
        nb_samples = len(phrases)
        print('Generating vectors for {} sentences using sent2vec model'.format(nb_samples))
        X_sent2vec = np.zeros((nb_samples, max_wordseq_len, word_dims), dtype=np.float32)
        for isent, (words1, words2, phrase1, phrase2) in enumerate(phrases):
            vectorize_words(words1, X_sent2vec, isent, word2vec)

        sent_vecs = sent2vec_model.predict(X_sent2vec, batch_size=batch_size, verbose=1)

        nx = len(sent_vecs)*len(sent_vecs[0])
        print("Computing the histo for {} x's".format(nx))
        xx = []
        for sent_vec in sent_vecs:
            for x in sent_vec:
                xx.append(x)

        nb_bins = 20
        x_hist, bin_edges = np.histogram(xx, bins=nb_bins)
        for ibin in range(nb_bins):
            print('[{}]\t{:<8.4f} .. {:<8.4f} => {}'.format(ibin, bin_edges[ibin], bin_edges[ibin+1], x_hist[ibin]))

        # нужно подобрать порог бинаризации так, чтобы обеспечивался заданный sparsity rate.
        x_min = min(xx)
        x_max = max(xx)
        nb_bins = 100
        required_sparsity = 0.1
        threshold = x_min
        x_hist, bin_edges = np.histogram(xx, bins=nb_bins)
        for i in range(1, len(bin_edges)):
            n1 = sum( x_hist[j] for j in range(i, nb_bins) )
            sparsity = float(n1) / nx
            if sparsity <= required_sparsity:
                threshold = bin_edges[i+1]
                print('threshold={} => sparsity={}'.format(threshold, sparsity))
                break


        # сохраним конфиг для модели sent2vec
        # sent2vec_dim
        sent2vec_config = {
         'max_wordseq_len': max_wordseq_len,
         'w2v_path': word2vector_path,
         'wordchar2vector_path': wordchar2vector_path,
         'PAD_WORD': PAD_WORD,
         'arch_filepath': arch_filepath2,
         'weights_path': weights_path2,
         'word_dims': word_dims,
         'sent2vec_dim': sent2vec_dim,
         'threshold': threshold
         }

        with open(os.path.join(tmp_folder, config_file_name2), 'w') as f:
            json.dump(sent2vec_config, f)

    else:
        os.remove(arch_filepath2)
        os.remove(weights_path2)

# </editor-fold>

# <editor-fold desc="evaluate">
if run_mode == 'evaluate':
    # Оценка качества натренированной модели на специальном наборе вопросов и ожидаемых выборов предпосылок
    # из тренировочного набора. Данная оценка кардинально отличается от валидации при тренировке модели,
    # так как показывает, насколько хорошо модель выбирает ПРАВИЛЬНУЮ предпосылку среди множества
    # альтернативных нерелевантных предпосылок, а не просто проверяет, что релевантность предпосылки
    # для вопроса > similarity_theshold.

    # Грузим проверочные вопросы для проверочных предпосылок из файла.
    # Формат такой:
    # T: правильная предпосылка
    # T: альтернативный вариант предпосылки
    # ...
    # Q: заданный вопрос №1
    # Q: заданный вопрос №2
    # ...
    # пустая строка
    # T: ...
    eval_data = EvaluationDataset(max_wordseq_len, tokenizer)
    eval_data.load(data_folder)

    word2vec = None
    word_dims = -1

    nb_good = 0
    nb_bad = 0

    for irecord, phrases in eval_data.generate_groups():
        if word2vec is None:
            word2vec, word_dims, w2v = load_word_vectors(wordchar2vector_path, word2vector_path)

        nb_samples = len(phrases)

        if False:
            X1_query = np.zeros((nb_samples, max_wordseq_len, word_dims), dtype=np.float32)
            X2_query = np.zeros((nb_samples, max_wordseq_len, word_dims), dtype=np.float32)

            for irow, (premise_words, question_words) in enumerate(phrases):
                vectorize_words(premise_words, X1_query, irow, word2vec)
                vectorize_words(question_words, X2_query, irow, word2vec)

            y_pred = model.predict(x={'input_words1': X1_query, 'input_words2': X2_query}, batch_size=batch_size, verbose=0)
        else:
            for input_xs in generate_rows(phrases, itertools.repeat(0, nb_samples), nb_samples, w2v, 2):
                #print('DEBUG @891 input_xs={}'.format(input_xs.keys()))
                y_pred = model.predict(x=input_xs)
                #print('DEBUG @893')
                break

        if False:
            # DEBUG START
            print('DEBUG START')

            print('predict: y_pred[0]={}'.format(y_pred[0][0]))
            print('predict: y_pred[1]={}'.format(y_pred[1][0]))

            for xgen in generate_rows(phrases, ys, 1, w2v, 2):
                x1_gen = xgen['input_words1']
                x2_gen = xgen['input_words2']

                print('x1_gen.shape]{}'.format(x1_gen.shape))
                with open('../tmp/x1_eval.txt', 'w') as wrt:
                    for istep in range(x1_gen.shape[1]):
                        for idim in range(x1_gen.shape[2]):
                            wrt.write('{:15e} '.format(x1_gen[0, istep, idim]))
                        wrt.write('\n')

                print('x2_gen.shape]{}'.format(x2_gen.shape))
                with open('../tmp/x2_eval.txt', 'w') as wrt:
                    for istep in range(x2_gen.shape[1]):
                        for idim in range(x2_gen.shape[2]):
                            wrt.write('{:15e} '.format(x2_gen[0, istep, idim]))
                        wrt.write('\n')

                break;

            y_pred = model.predict_generator(generator=generate_rows(phrases, ys, 1, w2v, 2),
                                             steps=1,
                                             verbose=0)

            print('y_pred[0]={}'.format(y_pred[0][0]))
            print('DEBUG END')
            exit(1)
            # DEBUG END

        # predict вернет список из списков, каждый длиной 1 элемент.
        # переформируем это безобразие в простой список чисел.
        y_pred = [y_pred[i][1] for i in range(len(y_pred))]

        # предпосылка с максимальной релевантностью
        max_index = np.argmax(y_pred)
        selected_premise = u' '.join(phrases[max_index][0]).strip()

        # эта выбранная предпосылка соответствует одному из вариантов
        # релевантных предпосылок в этой группе?
        if eval_data.is_relevant_premise(irecord, selected_premise):
            nb_good += 1
            print(EvaluationMarkup.ok_color + EvaluationMarkup.ok_bullet + EvaluationMarkup.close_color, end='')
        else:
            nb_bad += 1
            print(EvaluationMarkup.fail_color + EvaluationMarkup.fail_bullet + EvaluationMarkup.close_color, end='')

        max_sim = np.max(y_pred)

        question = phrases[0][1]
        print(u'{:<40} {:<40} {}/{}'.format(u' '.join(question), u' '.join(phrases[max_index][0]), y_pred[max_index], y_pred[0]))

        # для отладки: top релевантных вопросов
        if False:
            print(u'Most similar premises for question {}'.format(u' '.join(question)))
            yy = [(y_pred[i], i) for i in range(len(y_pred))]
            yy = sorted(yy, key=lambda z:-z[0])

            for sim, index in yy[:5]:
                print(u'{:.4f} {}'.format(sim, u' '.join(phrases[index][0])))

            exit(0)

    # Итоговая точность выбора предпосылки.
    accuracy = float(nb_good)/float(nb_good+nb_bad)
    print('accuracy={}'.format(accuracy))

# </editor-fold>

# <editor-fold desc="query">
if run_mode == 'query':
    # Ввод двух предложений с клавиатуры и выдача их релевантности.

    word2vec, word_dims, w2v = load_word_vectors(wordchar2vector_path, word2vector_path)

    X1_probe = np.zeros((1, max_wordseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((1, max_wordseq_len, word_dims), dtype=np.float32)

    while True:
        X1_probe.fill(0)
        X2_probe.fill(0)

        print('\nEnter two phrases:')
        phrase1 = raw_input('phrase #1 (premise):> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase1) == 0:
            break

        phrase2 = raw_input('phrase #2 (question):> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase2) == 0:
            break

        words1 = tokenizer.tokenize(phrase1)
        words2 = tokenizer.tokenize(phrase2)

        all_words_known = True
        for word in itertools.chain(words1, words2):
            if word not in word2vec:
                print(u'Unknown word {}'.format(word))
                all_words_known = False

        if all_words_known:
            vectorize_words(pad_wordseq(words1, max_wordseq_len), X1_probe, 0, word2vec)
            vectorize_words(pad_wordseq(words2, max_wordseq_len), X2_probe, 0, word2vec)
            y_probe = model.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})
            sim = y_probe[0][0]
            print('sim={}'.format(sim))

            if False:
                # содержимое X*_probe для отладки
                print('X1_probe.shape]{}'.format(X1_probe.shape))
                with open('../tmp/x1_query.txt', 'w') as wrt:
                    for istep in range(X1_probe.shape[1]):
                        for idim in range(X1_probe.shape[2]):
                            wrt.write('{:15e} '.format(X1_probe[0, istep, idim]))
                        wrt.write('\n')

                print('X2_probe.shape]{}'.format(X2_probe.shape))
                with open('../tmp/x2_query.txt', 'w') as wrt:
                    for istep in range(X2_probe.shape[1]):
                        for idim in range(X2_probe.shape[2]):
                            wrt.write('{:15e} '.format(X2_probe[0, istep, idim]))
                        wrt.write('\n')

# </editor-fold>

# <editor-fold desc="query2">
if run_mode == 'query2':
    # С клавиатуры задается путь к файлу с предложениями, и второе предложение.
    # Модель делает оценку релевантности для каждого предложения в файле и введенного предложения,
    # и сохраняет список оценок с сортировкой.

    word2vec, word_dims, w2v = load_word_vectors(wordchar2vector_path, word2vector_path)

    path1 = raw_input('path to text file with phrases:\n> ').decode(sys.stdout.encoding).strip().lower()

    phrases1 = []
    segm_mode = raw_input('Use EOL markers (1) or segmenter (2) to split file to sentences?').strip()

    max_nb_facts = int(raw_input('maximum number of samples to read from file (-1 means all):\n> ').strip())
    if max_nb_facts == -1:
        max_nb_facts = 10000000

    if segm_mode == 2:
        segmenter = Segmenter()
        phrases0 = segmenter.split(codecs.open(path1, 'r', 'utf-8').readlines())
        for phrase in enumerate(phrases):
            words = tokenizer.tokenize(phrase)
            if len(words) > 0:
                phrases1.append(words)
            if len(phrases1) >= max_nb_facts:
                break
    else:
        with codecs.open(path1, 'r', 'utf-8') as rdr:
            for phrase in rdr:
                words = tokenizer.tokenize(phrase)
                if len(words) > 0:
                    phrases1.append(words)
                if len(phrases1) >= max_nb_facts:
                    break

    nb_phrases = len(phrases1)
    print(u'{1} phrases are loaded from {0}'.format(path1, nb_phrases))

    X1_probe = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)

    for iphrase, words1 in enumerate(phrases1):
        vectorize_words(pad_wordseq(words1, max_wordseq_len), X1_probe, iphrase, word2vec)

    while True:
        # меняется только вторая матрица (с консоли вводится новый вопрос).
        X2_probe.fill(0)

        phrase2 = raw_input('phrase #2:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase2) == 0:
            break

        words2 = tokenizer.tokenize(phrase2)

        all_words_known = True
        for word in words2:
            if word not in word2vec:
                print(u'Unknown word {}'.format(word))
                all_words_known = False

        if all_words_known:
            # вторая матрица будет содержать множество повторов одной и той же фразы,
            # поэтому сформируем только первую строку, а остальные накопируем из нее.
            vectorize_words(pad_wordseq(words2, max_wordseq_len), X2_probe, 0, word2vec)
            for i in range(1, nb_phrases):
                X2_probe[i, :, :] = X2_probe[0, :, :]

            y_probe = model.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})

            sent_rels = [(phrases1[i], y_probe[i][0]) for i in range(nb_phrases)]
            sent_rels = sorted(sent_rels, key=lambda z:-z[1])

            # Выведем top N фраз из файла
            for phrase1, rel in sent_rels[0:10]:
                print(u'{:4f}\t{}'.format(rel, u' '.join(phrase1)))

            print('\n\n')

# </editor-fold>

# -*- coding: utf-8 -*-
"""
Модель для определения СИНОНИМИИ - семантической эквивалентности двух фраз,
включая позиционные и синтаксические перефразировки, лексические и фразовые
синонимы. В отличие от модели для РЕЛЕВАНТНОСТИ предпосылки и вопроса, в этой
модели предполагается, что объем информации в обеих фразах примерно одинаков,
то есть "кошка спит" и "черная кошка сладко спит" не считаются полными синонимами.

Вариант архитектуры с TRIPLE LOSS.

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
import operator
import math
import tqdm
import numpy as np
import pandas as pd

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


PHRASE_DIM = 64

nb_neg_per_posit = 1

padding = 'left'

random.seed(123456789)
np.random.seed(123456789)


class Sample3:
    def __init__(self, anchor, positive, negative):
        assert(len(anchor) > 0)
        assert(len(positive) > 0)
        self.anchor = anchor
        self.positive = positive
        self.negative = negative


def ngrams(s, n):
    return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1&shingles2))/float(1e-8+len(shingles1|shingles2))


def select_most_similar(phrase1, phrases2, topn):
    sims = [(phrase2, jaccard(phrase1, phrase2, 3)) for phrase2 in phrases2]
    sims = sorted(sims, key=lambda z: -z[1])
    return list(map(operator.itemgetter(0), sims[:topn]))


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
    X3_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)

    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    while True:
        for irow, sample in enumerate(samples):
            vectorize_words(sample.anchor_words, X1_batch, batch_index, w2v)
            vectorize_words(sample.positive_words, X2_batch, batch_index, w2v)
            vectorize_words(sample.negative_words, X3_batch, batch_index, w2v)

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                xx = {'input_anchor': X1_batch,
                      'input_positive': X2_batch,
                      'input_negative': X3_batch}
                if mode == 1:
                    yield (xx, {'output': y_batch})
                else:
                    yield xx

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                X3_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


def triplet_loss(y_true, y_pred):
    """
    See also https://stackoverflow.com/questions/38260113/implementing-contrastive-loss-and-triplet-loss-in-tensorflow

    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    alpha = 0.5

    anchor   = y_pred[:, 0:PHRASE_DIM]
    positive = y_pred[:, PHRASE_DIM:2*PHRASE_DIM]
    negative = y_pred[:, 2*PHRASE_DIM:3*PHRASE_DIM]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    loss = K.mean(loss, axis=-1)
    return loss


# -------------------------------------------------------------------


NB_EPOCHS = 1000

# Разбор параметров тренировки в командной строке
parser = argparse.ArgumentParser(description='Neural model for short text synonymy')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query | query2')
parser.add_argument('--arch', type=str, default='lstm', help='neural model architecture: lstm | lstm(cnn)')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/synonymy_dataset3.csv', help='path to input dataset with triplets')
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

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_synonymy_tripleloss.log'))

run_mode = args.run_mode
# Варианты значения для параметра run_mode:
# run_mode='train'
# Тренировка модели по заранее приготовленному датасету с парами предложений
# run_mode = 'query'
# В консоли вводятся два предложения, модель оценивает их релевантность.


config_path = os.path.join(tmp_folder, 'nn_synonymy_tripleloss.config')
arch_filepath = os.path.join(tmp_folder, 'nn_synonymy_tripleloss.arch')
weights_path = os.path.join(tmp_folder, 'nn_synonymy_tripleloss.weights')

tokenizer = Tokenizer()

if run_mode == 'train':
    logging.info('Start with run_mode==train')

    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
    word_dims = embeddings.vector_size

    samples3 = []
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    for anchor, positive, negative in itertools.izip(df['anchor'].values, df['positive'].values, df['negative'].values):
        samples3.append(Sample3(anchor, positive, negative))

    max_wordseq_len = 0
    for sample in samples3:
        for phrase in [sample.anchor, sample.positive, sample.negative]:
            words = tokenizer.tokenize(phrase)
            max_wordseq_len = max(max_wordseq_len, len(words))

    logging.info('max_wordseq_len={}'.format(max_wordseq_len))

    if padding == 'left':
        for sample in samples3:
            sample.anchor_words = lpad_wordseq(tokenizer.tokenize(sample.anchor), max_wordseq_len)
            sample.positive_words = lpad_wordseq(tokenizer.tokenize(sample.positive), max_wordseq_len)
            sample.negative_words = lpad_wordseq(tokenizer.tokenize(sample.negative), max_wordseq_len)
    else:
        for sample in samples3:
            sample.anchor_words = rpad_wordseq(tokenizer.tokenize(sample.anchor), max_wordseq_len)
            sample.positive_words = rpad_wordseq(tokenizer.tokenize(sample.positive), max_wordseq_len)
            sample.negative_words = rpad_wordseq(tokenizer.tokenize(sample.negative), max_wordseq_len)

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'model': 'nn_synonymy_tripleloss',
        'max_wordseq_len': max_wordseq_len,
        'w2v_path': word2vector_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'PHRASE_DIM': PHRASE_DIM,
        'padding': padding,
        'arch_filepath': arch_filepath,
        'weights_path': weights_path,
        'word_dims': word_dims,
        'net_arch': net_arch,
    }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    logging.info('Constructing the NN model arch={}...'.format(net_arch))

    input_anchor = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_anchor')
    input_positive = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_positive')
    input_negative = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_negative')

    if net_arch == 'lstm':
        lstm = recurrent.LSTM(PHRASE_DIM, return_sequences=False, name='anchor_flow')
        anchor_flow = lstm(input_anchor)
        positive_flow = lstm(input_positive)
        negative_flow = lstm(input_negative)

        #dense = Dense(units=PHRASE_DIM, activation='relu')
        #anchor_flow = dense(anchor_flow)
        #positive_flow = dense(positive_flow)
        #negative_flow = dense(negative_flow)


    if net_arch == 'lstm(cnn)':
        nb_filters = 128  # 128
        rnn_size = word_dims * 2

        layers1 = []  # anchor
        layers2 = []  # positive
        layers3 = []  # negative
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

            # поверх сверточных идут рекуррентные слои
            lstm = recurrent.LSTM(rnn_size, return_sequences=False)

            conv_layer1 = conv(input_anchor)
            conv_layer1 = pooler(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            layers1.append(conv_layer1)

            conv_layer2 = conv(input_positive)
            conv_layer2 = pooler(conv_layer2)
            conv_layer2 = lstm(conv_layer2)
            layers2.append(conv_layer2)

            conv_layer3 = conv(input_negative)
            conv_layer3 = pooler(conv_layer3)
            conv_layer3 = lstm(conv_layer3)
            layers3.append(conv_layer3)

        anchor_flow = keras.layers.concatenate(inputs=layers1)
        positive_flow = keras.layers.concatenate(inputs=layers2)
        negative_flow = keras.layers.concatenate(inputs=layers3)

        # полносвязный слой приводит размерность вектора предложения к заданной
        dense = Dense(units=PHRASE_DIM, activation='sigmoid')
        anchor_flow = dense(anchor_flow)
        positive_flow = dense(positive_flow)
        negative_flow = dense(negative_flow)

    output = keras.layers.concatenate(inputs=[anchor_flow, positive_flow, negative_flow], name='output')

    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=output)
    model.compile(loss=triplet_loss, optimizer='nadam')
    model.summary()

    # разбиваем датасет на обучающую, валидационную и оценочную части.
    # таким образом, финальная оценка будет идти по сэмплам, которые
    # никак не участвовали в обучении модели, в том числе через early stopping
    SEED = 123456
    TEST_SHARE = 0.3
    train_samples, val_samples = train_test_split(samples3, test_size=TEST_SHARE, random_state=SEED)
    val_samples, eval_samples = train_test_split(val_samples, test_size=0.3, random_state=SEED)
    logging.info('train_samples.count={}'.format(len(train_samples)))
    logging.info('val_samples.count={}'.format(len(val_samples)))
    logging.info('eval_samples.count={}'.format(len(eval_samples)))

    # начало отладки - прогоним через необученную модель данные, сделаем расчет loss'а
    #y_pred = model.predict_generator(generate_rows(train_samples, batch_size, embeddings, 1), steps=1)
    #for i, y in enumerate(y_pred):
    #    loss = triplet_loss2(y)
    #    print('{} loss={}'.format(i, loss))

    logging.info('Start training...')
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=20,
                                   verbose=1,
                                   mode='auto')

    nb_validation_steps = len(val_samples)//batch_size

    hist = model.fit_generator(generator=generate_rows(train_samples, batch_size, embeddings, 1),
                               steps_per_epoch=len(train_samples)//batch_size,
                               epochs=NB_EPOCHS,
                               verbose=1,
                               callbacks=[model_checkpoint, early_stopping],
                               validation_data=generate_rows(val_samples, batch_size, embeddings, 1),
                               validation_steps=nb_validation_steps,
                               )

    logging.info('Estimating final accuracy on eval dataset...')

    # загрузим чекпоинт с оптимальными весами
    model.load_weights(weights_path)

    # Создадим сетку, которая будет выдавать вектор одной заданной фразы
    model = Model(inputs=input_anchor, outputs=anchor_flow)
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    # Сохраняем архитектуру и веса этой сетки.
    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    model.save_weights(weights_path)

    # Генерируем векторы для предложений из валидационного набора
    val_phrases = set()
    for sample in eval_samples:
        val_phrases.add(sample.anchor)
        val_phrases.add(sample.positive)
        val_phrases.add(sample.negative)

    val_phrases = [(s, tokenizer.tokenize(s)) for s in val_phrases]
    logging.info('Vectorization of {} phrases'.format(len(val_phrases)))

    nb_phrases = len(val_phrases)

    X_data = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
    for iphrase, phrase in enumerate(val_phrases):
        vectorize_words(phrase[1], X_data, iphrase, embeddings)

    y_pred = model.predict(x=X_data, batch_size=batch_size, verbose=1)

    # Теперь для каждой фразы мы знаем вектор
    phrase2v = dict()
    for i in range(nb_phrases):
        phrase = val_phrases[i][0]
        v = y_pred[i]
        phrase2v[phrase] = v


    nb_error = 0
    nb_good = 0
    for sample in eval_samples:
        anchor = phrase2v[sample.anchor]
        positive = phrase2v[sample.positive]
        negative = phrase2v[sample.negative]

        # distance between the anchor and the positive
        pos_dist = np.sum(np.square(anchor - positive))

        # distance between the anchor and the negative
        neg_dist = np.sum(np.square(anchor - negative))

        if pos_dist < neg_dist:
            nb_good += 1
        else:
            nb_error += 1

    acc = nb_good/float(nb_error+nb_good)
    logging.info('Validation results: accuracy={}'.format(acc))

# </editor-fold>

if run_mode == 'query2':
    # В консоли вводится предложение, для которого в списке smalltalk.txt
    # ищутся ближайшие.

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_wordseq_len = model_config['max_wordseq_len']
        w2v_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        net_arch = model_config['net_arch']

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

        all_phrases = list(itertools.chain(phrases2, [phrase1]))
        nb_phrases = len(all_phrases)

        print('Vectorization of {} samples'.format(nb_phrases))
        X_data = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
        for iphrase, phrase in enumerate(all_phrases):
            words = tokenizer.tokenize(phrase)
            vectorize_words(words, X_data, iphrase, embeddings)

        y_pred = model.predict(x=X_data, batch_size=batch_size, verbose=1)

        # Теперь для каждой фразы мы знаем вектор
        phrase2v = dict()
        for i in range(nb_phrases):
            phrase = all_phrases[i]
            v = y_pred[i]
            phrase2v[phrase] = v

        # Можем оценить близость введенной фразы к каждому образцу в smalltalk.txt
        phrase_sims = []
        v1 = phrase2v[phrase1]
        for phrase2 in phrases2:
            v2 = phrase2v[phrase2]
            dist = np.sum(np.square(v1 - v2))
            sim = math.exp(-dist)
            phrase_sims.append((phrase2, sim))

        phrase_sims = sorted(phrase_sims, key=lambda z: -z[1])
        for phrase, sim in phrase_sims[:10]:
            print(u'{:6.4f} {}'.format(sim, phrase))

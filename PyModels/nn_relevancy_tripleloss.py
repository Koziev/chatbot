# -*- coding: utf-8 -*-
"""

Тренировка нейросетевой модели для определения РЕЛЕВАНТНОСТИ ПРЕДПОСЫЛКИ И ВОПРОСА.
Вариант архитектуры с TRIPLE LOSS.
Для проекта чатбота https://github.com/Koziev/chatbot

См. также:
"Deep Semantic Similarity Model" https://www.microsoft.com/en-us/research/project/dssm/
"FaceNet: A Unified Embedding for Face Recognition and Clustering" https://arxiv.org/pdf/1503.03832v3.pdf
"Lossless triplet loss" https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
"Similarity learning"  https://en.wikipedia.org/wiki/Similarity_learning
"""

from __future__ import division
from __future__ import print_function

import codecs
import itertools
import json
import os
import argparse
import random
import logging
import operator
import numpy as np
import pandas as pd

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.merge import add, multiply
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
import keras.regularizers

import sklearn.metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from utils.tokenizer import Tokenizer
from utils.padding_utils import lpad_wordseq, rpad_wordseq
from utils.padding_utils import PAD_WORD
from trainers.word_embeddings import WordEmbeddings
import utils.console_helpers
import utils.logging_helpers
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup


# Длина вектора предложения
phrase_dim = 200

# Предложения приводятся к единой длине путем добавления слева или справа
# пустых токенов. Параметр padding определяет режим выравнивания.
padding = 'right'

PRETRAIN = False

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
    return float(len(shingles1 & shingles2)) / float(1e-8 + len(shingles1 | shingles2))


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
        # эксперимент - сортируем сэмплы по тексту anchor'а
        # samples = sorted(samples, key=lambda s: s.anchor)

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

    anchor = y_pred[:, 0:phrase_dim]
    positive = y_pred[:, phrase_dim:2 * phrase_dim]
    negative = y_pred[:, 2 * phrase_dim:3 * phrase_dim]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    loss = K.mean(loss, axis=-1)
    return loss


def triplet_loss2(y_true, y_pred):
    """Модификация triplet_loss с cos-метрикой"""
    alpha = 0.5

    anchor = y_pred[:, 0:phrase_dim]
    positive = y_pred[:, phrase_dim:2 * phrase_dim]
    negative = y_pred[:, 2 * phrase_dim:3 * phrase_dim]

    # distance between the anchor and the positive
    pos_dist = -K.sum(anchor * positive, axis=-1)

    # distance between the anchor and the negative
    neg_dist = -K.sum(anchor * negative, axis=-1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    loss = K.mean(loss, axis=-1)
    return loss


def lossless_triplet_loss(y_true, y_pred, N=phrase_dim, beta=phrase_dim, epsilon=1e-8):
    """
    Implementation of the triplet loss function
    https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)


    Returns:
    loss -- real number, value of the loss
    """

    anchor = y_pred[:, 0:phrase_dim]
    positive = y_pred[:, phrase_dim:2 * phrase_dim]
    negative = y_pred[:, 2 * phrase_dim:3 * phrase_dim]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # Non Linear Values

    # -ln(-x/N+1)
    pos_dist = -K.log(-pos_dist / beta + 1 + epsilon)
    neg_dist = -K.log(-(N - neg_dist) / beta + 1 + epsilon)

    # compute loss
    loss = neg_dist + pos_dist

    return loss


def convert_samples2(samples, phrase2v):
    added = set()
    samples2 = []
    for sample in samples:
        pair1 = sample.anchor + '|' + sample.positive
        if pair1 not in added:
            samples2.append((phrase2v[sample.anchor], phrase2v[sample.positive], 1))
            added.add(pair1)

        pair2 = sample.anchor + '|' + sample.negative
        if pair2 not in added:
            samples2.append((phrase2v[sample.anchor], phrase2v[sample.negative], 0))
            added.add(pair2)

    return samples2


def generate_rows2(samples, batch_size, mode):
    if mode == 1:
        # При обучении сетки каждую эпоху тасуем сэмплы.
        random.shuffle(samples)

    phrase_dim = len(samples[0][0])

    batch_index = 0
    batch_count = 0

    X1_batch = np.zeros((batch_size, phrase_dim), dtype=np.float32)
    X2_batch = np.zeros((batch_size, phrase_dim), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    while True:
        for irow, sample in enumerate(samples):
            X1_batch[batch_index] = sample[0]
            X2_batch[batch_index] = sample[1]
            y_batch[batch_index, sample[2]] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                xx = {'input1': X1_batch,
                      'input2': X2_batch}
                if mode == 1:
                    yield (xx, {'output': y_batch})
                else:
                    yield xx

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0

# -------------------------------------------------------------------


class ModelCheckpoint_Acc(keras.callbacks.Callback):
    def __init__(self, X_val, val_phrases, val_samples, model, weights_path):
        self.epoch = 0
        self.X_val = X_val
        self.model = model
        self.val_phrases = val_phrases
        self.val_samples = val_samples
        self.best_val_acc = -np.inf  # для сохранения самой точной модели
        self.weights_path = weights_path
        self.wait = 0  # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 20

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1

        y_pred = self.model.predict(x=self.X_val, batch_size=batch_size, verbose=0)

        # Теперь для каждой фразы мы знаем вектор
        phrase2v = dict()
        nb_phrases = len(self.val_phrases)
        for i in range(nb_phrases):
            phrase = self.val_phrases[i][0]
            v = y_pred[i]
            phrase2v[phrase] = v

        nb_error = 0
        nb_good = 0
        for sample in self.val_samples:
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

        val_acc = nb_good / float(nb_error + nb_good)

        if val_acc > self.best_val_acc:
            utils.console_helpers.print_green_line('\nInstance accuracy improved from {} to {}, saving model to {}\n'.format(self.best_val_acc, val_acc, self.weights_path))
            self.best_val_acc = val_acc
            self.model.save_weights(self.weights_path)
            self.wait = 0
        else:
            print('\nTotal instance accuracy={} did not improve (current best acc={})\n'.format(val_acc, self.best_val_acc))
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_best_accuracy(self):
        return self.best_val_acc

    def new_epochs(self):
        self.wait = 0
        self.model.stop_training = False

# -------------------------------------------------------------------


NB_EPOCHS = 1000

# Разбор параметров тренировки в командной строке
parser = argparse.ArgumentParser(description='Neural model for premise-question relevancy')
parser.add_argument('--run_mode', type=str, default='train', choices='train query query2 evaluate'.split(), help='what to do')
parser.add_argument('--arch', type=str, default='lstm', choices='lstm lstm(cnn) lstm(lstm) bilstm'.split(), help='neural model architecture')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--phrase_dim', type=int, default=64, help='Sentence embedding vector size')
parser.add_argument('--input', type=str, default='../data/relevancy_dataset3.csv', help='path to input dataset with triplets')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.bin', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')
parser.add_argument('--train_model1', type=bool, default=True, help='train sentence embedding model')
parser.add_argument('--train_model2', type=bool, default=False, help='train metric model')

args = parser.parse_args()
data_folder = args.data_dir
input_path = args.input
tmp_folder = args.tmp
train_model1 = args.train_model1
train_model2 = args.train_model2
phrase_dim = args.phrase_dim


wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)
batch_size = args.batch_size
net_arch = args.arch

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_relevancy_tripleloss.log'))

run_mode = args.run_mode
# Варианты значения для параметра run_mode:
# 'train' - Тренировка модели по заранее приготовленному датасету с парами предложений
# 'query' - В консоли вводятся два предложения, модель оценивает их релевантность.
# 'query2' - В вонсоли вводится вопрос, из текстового файла загружаются варианты предпосылок, релевантность к которым выдается.


config_path = os.path.join(tmp_folder, 'nn_relevancy_tripleloss.config')
arch_filepath = os.path.join(tmp_folder, 'nn_relevancy_tripleloss.arch')
weights_path0 = os.path.join(tmp_folder, 'nn_relevancy_tripleloss0.weights')
weights_path = os.path.join(tmp_folder, 'nn_relevancy_tripleloss.weights')
weights_path2 = os.path.join(tmp_folder, 'nn_relevancy2.weights')

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

    pad_func = lpad_wordseq if padding == 'left' else rpad_wordseq

    for sample in samples3:
        sample.anchor_words = pad_func(tokenizer.tokenize(sample.anchor), max_wordseq_len)
        sample.positive_words = pad_func(tokenizer.tokenize(sample.positive), max_wordseq_len)
        sample.negative_words = pad_func(tokenizer.tokenize(sample.negative), max_wordseq_len)

    if PRETRAIN:
        logging.info('Building samples for pretraining')
        pretrain_samples = []
        anchors = [sample.anchor for sample in samples3]
        positives = [sample.positive for sample in samples3]
        negatives = [sample.negative for sample in samples3]

        for _ in range(1000000):
            sample = Sample3(random.choice(anchors),
                             random.choice(positives),
                             random.choice(negatives))

            sample.anchor_words = pad_func(tokenizer.tokenize(sample.anchor), max_wordseq_len)
            sample.positive_words = pad_func(tokenizer.tokenize(sample.positive), max_wordseq_len)
            sample.negative_words = pad_func(tokenizer.tokenize(sample.negative), max_wordseq_len)

            pretrain_samples.append(sample)

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'model': 'nn_relevancy_tripleloss',
        'max_wordseq_len': max_wordseq_len,
        'w2v_path': word2vector_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'PHRASE_DIM': phrase_dim,
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
        lstm = recurrent.LSTM(phrase_dim,
                              return_sequences=False,
                              name='anchor_flow',
                              go_backwards=True)
        anchor_flow = lstm(input_anchor)
        positive_flow = lstm(input_positive)
        negative_flow = lstm(input_negative)

    if net_arch == 'bilstm':
        lstm = recurrent.LSTM(phrase_dim,
                              return_sequences=False,
                              name='anchor_flow',
                              go_backwards=True)
        lstm = Bidirectional(lstm)
        anchor_flow = lstm(input_anchor)
        positive_flow = lstm(input_positive)
        negative_flow = lstm(input_negative)

        # dense = Dense(units=PHRASE_DIM, activation='sigmoid', activity_regularizer=keras.regularizers.l1(0.00001))
        # anchor_flow = dense(anchor_flow)
        # anchor_flow = Dropout(rate=0.2)(anchor_flow)
        # positive_flow = dense(positive_flow)
        # positive_flow = Dropout(rate=0.2)(positive_flow)
        # negative_flow = dense(negative_flow)
        # negative_flow = Dropout(rate=0.2)(negative_flow)

    if net_arch == 'lstm(lstm)':
        lstm = recurrent.LSTM(phrase_dim, return_sequences=True)
        anchor_flow = lstm(input_anchor)
        positive_flow = lstm(input_positive)
        negative_flow = lstm(input_negative)

        lstm2 = recurrent.LSTM(phrase_dim, return_sequences=False, name='anchor_flow')
        anchor_flow = lstm2(anchor_flow)
        positive_flow = lstm2(positive_flow)
        negative_flow = lstm2(negative_flow)

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
            # pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')

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
        dense = Dense(units=phrase_dim, activation='sigmoid')
        anchor_flow = dense(anchor_flow)
        positive_flow = dense(positive_flow)
        negative_flow = dense(negative_flow)

    output = keras.layers.concatenate(inputs=[anchor_flow, positive_flow, negative_flow], name='output')

    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=output)
    model.compile(loss=triplet_loss, optimizer='nadam')
    # model.compile(loss=triplet_loss, optimizer='sgd')
    model.summary()

    keras.utils.plot_model(model,
                           to_file=os.path.join(tmp_folder, 'nn_relevancy_tripletloss.arch.png'),
                           show_shapes=False,
                           show_layer_names=True)

    # Создадим сетку, которая будет выдавать вектор одной заданной фразы
    model1 = Model(inputs=input_anchor, outputs=anchor_flow)
    model1.compile(loss='mse', optimizer='adam')
    model1.summary()

    # Сохраняем архитектуру этой сетки.
    with open(arch_filepath, 'w') as f:
        f.write(model1.to_json())

    if PRETRAIN:
        logging.info('Start pretraining...')
        hist = model.fit_generator(generator=generate_rows(pretrain_samples, batch_size, embeddings, 1),
                                   steps_per_epoch=len(pretrain_samples) // batch_size,
                                   epochs=20,
                                   verbose=1)
        logging.info('End of pretraining.')

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
    # y_pred = model.predict_generator(generate_rows(train_samples, batch_size, embeddings, 1), steps=1)
    # for i, y in enumerate(y_pred):
    #     loss = triplet_loss2(y)
    #     print('{} loss={}'.format(i, loss))

    # Валидационный набор
    eval_phrases = set()
    for sample in eval_samples:
        eval_phrases.add(sample.anchor)
        eval_phrases.add(sample.positive)
        eval_phrases.add(sample.negative)

    eval_phrases = [(s, tokenizer.tokenize(s)) for s in eval_phrases]
    logging.info('Vectorization of {} phrases for validation'.format(len(eval_phrases)))

    nb_phrases = len(eval_phrases)
    X_eval = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
    for iphrase, phrase in enumerate(eval_phrases):
        vectorize_words(phrase[1], X_eval, iphrase, embeddings)

    logging.info('Start training...')
    model_checkpoint = ModelCheckpoint(weights_path0, monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=20,
                                   verbose=1,
                                   mode='auto')

    if train_model1:
        hist = model.fit_generator(generator=generate_rows(train_samples, batch_size, embeddings, 1),
                                   steps_per_epoch=len(train_samples) // batch_size,
                                   epochs=NB_EPOCHS,
                                   verbose=1,
                                   callbacks=[model_checkpoint, early_stopping],
                                   validation_data=generate_rows(val_samples, batch_size, embeddings, 1),
                                   validation_steps=len(val_samples) // batch_size,
                                   )
    # best_acc = model_checkpoint.get_best_accuracy()
    # logging.info('Best validation accuracy={}'.format(best_acc))

    model.load_weights(weights_path0)

    model1.save_weights(weights_path)

    y_pred = model1.predict(x=X_eval, batch_size=batch_size, verbose=0)

    # Теперь для каждой фразы мы знаем вектор
    phrase2v = dict()
    nb_phrases = len(eval_phrases)
    for i in range(nb_phrases):
        phrase = eval_phrases[i][0]
        v = y_pred[i]
        phrase2v[phrase] = v

    # первая оценка точности берет каждую тройку и смотрит,
    # будет ли позивтивный вектор ближе к опорному, чем негативный.
    # если будет - значит классификация правильная, иначе - неправильная.
    # Отсюда получается доля верных классификаций - точность №1.
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

    acc = nb_good / float(nb_error + nb_good)
    logging.info('Validation results: inner accuracy of ranking in triplets = {}'.format(acc))

    # Точность ранжирования внутри троек не очень информативна как мера точности
    # классификатора при выборе одной предпосылки среди множества потенциальных предпосылок.
    # Сделаем вторую оценку, выделив пары (anchor, positive) и (anchor, negative) из оценочных
    # сэмплов.
    y2_pred = []
    y2_true = []
    eval_samples2 = []
    all_evals = set()

    for sample in eval_samples:
        anchor = phrase2v[sample.anchor]
        positive = phrase2v[sample.positive]
        negative = phrase2v[sample.negative]

        k = sample.anchor + '|' + sample.positive
        if k not in all_evals:
            all_evals.add(k)

            # cosine similarity between the anchor and the positive
            pos_dist = v_cosine(anchor, positive)

            y2_pred.append(pos_dist)
            y2_true.append(1.0)

        k = sample.anchor + '|' + sample.negative
        if k not in all_evals:
            all_evals.add(k)

            # cosine similarity between the anchor and the negative
            neg_dist = v_cosine(anchor, negative)

            y2_pred.append(neg_dist)
            y2_true.append(0.0)

    aucroc = sklearn.metrics.roc_auc_score(y2_true, y2_pred)
    logging.info('Validation results: auc roc={}'.format(aucroc))

    if True:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y2_true, y2_pred)
        roc_auc = sklearn.metrics.auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = {:0.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        #ax2 = plt.gca().twinx()
        #ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
        #ax2.set_ylabel('Threshold', color='r')
        #ax2.set_ylim([thresholds[-1], thresholds[0]])
        #ax2.set_xlim([fpr[0], fpr[-1]])

        plt.savefig(os.path.join(tmp_folder, 'nn_relevancy_tripleloss.roc_and_threshold.png'))
        plt.close()


    if train_model2:
        # Теперь вторая модель поверх векторов из tripleloss модели

        all_phrases = set()
        for sample in samples3:
            all_phrases.add(sample.anchor)
            all_phrases.add(sample.positive)
            all_phrases.add(sample.negative)

        all_phrases = [(s, tokenizer.tokenize(s)) for s in all_phrases]
        nb_phrases = len(all_phrases)
        X_all = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
        for iphrase, phrase in enumerate(all_phrases):
            vectorize_words(phrase[1], X_all, iphrase, embeddings)

        y_pred = model1.predict(x=X_all, batch_size=batch_size, verbose=1)

        # Теперь для каждой фразы мы знаем вектор
        phrase2v = dict()
        nb_phrases = len(all_phrases)
        for i in range(nb_phrases):
            phrase = all_phrases[i][0]
            v = y_pred[i]
            phrase2v[phrase] = v

        input1 = Input(shape=(phrase_dim,), dtype='float32', name='input1')
        input2 = Input(shape=(phrase_dim,), dtype='float32', name='input2')

        if False:
            net2 = keras.layers.concatenate([input1, input2])
            net2 = Dense(units=phrase_dim, activation='relu')(net2)
        else:
            addition = add([input1, input2])
            minus_y1 = Lambda(lambda x: -x, output_shape=(phrase_dim,))(input1)
            mul = add([input2, minus_y1])
            mul = multiply([mul, mul])
            net2 = keras.layers.concatenate(inputs=[mul, addition, input1, input2])

        net2 = Dense(units=phrase_dim // 2, activation='relu')(net2)
        net2 = Dense(units=2, activation='softmax', name='output')(net2)

        model2 = Model(inputs=[input1, input2], outputs=net2)
        model2.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer='nadam',
                       metrics=['accuracy'])
        model2.summary()

        logging.info('Start training model2...')
        model_checkpoint = ModelCheckpoint(weights_path2, monitor='val_acc',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='auto')
        early_stopping = EarlyStopping(monitor='val_acc',
                                       patience=20,
                                       verbose=1,
                                       mode='auto')

        train_samples2 = convert_samples2(train_samples, phrase2v)
        val_samples2 = convert_samples2(val_samples, phrase2v)

        hist = model2.fit_generator(generator=generate_rows2(train_samples2, batch_size, 1),
                                    steps_per_epoch=len(train_samples2) // batch_size,
                                    epochs=NB_EPOCHS,
                                    verbose=1,
                                    callbacks=[model_checkpoint, early_stopping],
                                    validation_data=generate_rows2(val_samples2, batch_size, 1),
                                    validation_steps=len(val_samples2) // batch_size,
                                    )

# </editor-fold>


if run_mode == 'query2':
    # В консоли вводится предложение, для которого в списках premises*.txt
    # ищутся ближайшие.

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_wordseq_len = model_config['max_wordseq_len']
        word2vector_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        net_arch = model_config['net_arch']
        padding = model_config['padding']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
    word_dims = embeddings.vector_size

    phrases2 = set()
    for fname in ['test_premises.txt']:
        with codecs.open(os.path.join(data_folder, fname), 'r', 'utf-8') as rdr:
            for line in rdr:
                phrase = line.strip()
                if len(phrase) > 5:
                    phrase = u' '.join(tokenizer.tokenize(phrase))
                    phrases2.add(phrase)

    phrases2 = list(phrases2)

    while True:
        phrase1 = utils.console_helpers.input_kbd('question:> ').strip().lower()
        if len(phrase1) == 0:
            break

        all_phrases = list(itertools.chain(phrases2, [phrase1]))
        nb_phrases = len(all_phrases)
        X_data = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
        pad_func = lpad_wordseq if padding == 'left' else rpad_wordseq

        for iphrase, phrase in enumerate(all_phrases):
            words = pad_func(tokenizer.tokenize(phrase), max_wordseq_len)
            vectorize_words(words, X_data, iphrase, embeddings)

        y_pred = model.predict(x=X_data, batch_size=batch_size, verbose=0)

        # Теперь для каждой фразы мы знаем вектор
        phrase2v = dict()
        for i in range(nb_phrases):
            phrase = all_phrases[i]
            v = y_pred[i]
            phrase2v[phrase] = v

        # Можем оценить близость введенной фразы к каждой предпосылке
        phrase_rels = []
        v1 = phrase2v[phrase1]

        if False:
            # Евклидово расстояние
            for phrase2 in phrases2:
                v2 = phrase2v[phrase2]
                dist = np.sum(np.square(v1 - v2))
                rel = -dist
                phrase_rels.append((phrase2, rel))
        else:
            # Косинусная мера
            for phrase2 in phrases2:
                v2 = phrase2v[phrase2]
                rel = v_cosine(v1, v2)
                phrase_rels.append((phrase2, rel))

        # Сортируем в порядке убывания похожести, выводим top лучших пар
        phrase_rels = sorted(phrase_rels, key=lambda z: -z[1])
        for phrase, rel in phrase_rels[:10]:
            print(u'{:6.4f} {}'.format(rel, phrase))


# <editor-fold desc="evaluate">
if run_mode == 'evaluate':
    # Оценка качества натренированной модели на специальном наборе вопросов и ожидаемых выборов предпосылок
    # из тренировочного набора.

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_wordseq_len = model_config['max_wordseq_len']
        word2vector_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        net_arch = model_config['net_arch']
        padding = model_config['padding']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    # Загружаем проверочные данные - предпосылки и вопросы
    eval_data = EvaluationDataset(max_wordseq_len, tokenizer, padding)
    eval_data.load(data_folder)

    nb_good = 0  # попадание предпосылки в top-1
    nb_good5 = 0
    nb_good10 = 0
    nb_total = 0

    word2vec = None

    for irecord, phrases in eval_data.generate_groups():
        if word2vec is None:
            # ленивая загрузка для удобства отладки.
            word2vec = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
            word_dims = word2vec.vector_size

        # phrases это список пар из предложений.
        # развернем его в список предложений, так как наша модель
        # умеет только получать вектор для одного предложения
        phrases1 = set()
        for phrase1, phrase2 in phrases:
            phrases1.add(phrase1)
            phrases1.add(phrase2)
        phrases1 = list(phrases1)

        nb_phrases = len(phrases1)

        X_data = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
        pad_func = lpad_wordseq if padding == 'left' else rpad_wordseq

        for iphrase, phrase in enumerate(phrases1):
            words = pad_func(tokenizer.tokenize(phrase), max_wordseq_len)
            vectorize_words(words, X_data, iphrase, word2vec)

        y_pred = model.predict(x=X_data, batch_size=batch_size, verbose=0)

        # Теперь для каждой фразы мы знаем вектор
        phrase2v = dict()
        for i in range(nb_phrases):
            phrase = phrases1[i]
            v = y_pred[i]
            phrase2v[phrase] = v

        # найдем пару с максимальной релевантностью
        selected_premise = None
        max_index = -1
        max_rel = 0.
        phrase_rels = []
        for iphrase, (phrase1, phrase2) in enumerate(phrases):
            v1 = phrase2v[phrase1]
            v2 = phrase2v[phrase2]
            if False:
                # Евклидово расстояние
                dist = np.sum(np.square(v1 - v2))
                rel = -dist  # math.exp(-dist)
            else:
                # Косинусная мера
                rel = v_cosine(v1, v2)
            phrase_rels.append(rel)
            if rel > max_rel:
                max_rel = rel
                max_index = iphrase
                selected_premise = phrase1

        nb_total += 1

        # эта выбранная предпосылка соответствует одному из вариантов
        # релевантных предпосылок в этой группе?
        if eval_data.is_relevant_premise(irecord, selected_premise):
            nb_good += 1
            nb_good5 += 1
            nb_good10 += 1
            print(EvaluationMarkup.ok_color + EvaluationMarkup.ok_bullet + EvaluationMarkup.close_color, end='')
        else:
            print(EvaluationMarkup.fail_color + EvaluationMarkup.fail_bullet + EvaluationMarkup.close_color, end='')

            # среди top-5 или top-10 предпосылок есть верная?
            sorted_phrases = [x for (x, _) in sorted(itertools.izip(phrases, phrase_rels), key=lambda z: -z[1])]

            for i in range(1, 10):
                selected_premise = sorted_phrases[i][0]
                if eval_data.is_relevant_premise(irecord, selected_premise):
                    if i < 5:
                        nb_good5 += 1  # верная предпосылка вошла в top-5
                    if i < 10:
                        nb_good10 += 1
                    break

        question = phrases[0][1]
        print(u'{:<60} {:<60} {}/{}'.format(question, phrases[max_index][0], phrase_rels[max_index], phrase_rels[0]))

    # Итоговая точность выбора предпосылки.
    accuracy = float(nb_good) / float(nb_total)
    print('accuracy       ={}'.format(accuracy))

    # Также выведем точность попадания верной предпосылки в top-5 и top-10
    accuracy5 = float(nb_good5) / float(nb_total)
    print('accuracy top-5 ={}'.format(accuracy5))

    accuracy10 = float(nb_good10) / float(nb_total)
    print('accuracy top-10={}'.format(accuracy10))
# </editor-fold>

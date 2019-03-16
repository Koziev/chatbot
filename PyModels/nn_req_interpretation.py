# -*- coding: utf-8 -*-
"""
Модель для определения необходимости интерпретации для реплики пользователя.
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


padding = 'left'

random.seed(123456789)
np.random.seed(123456789)


class Sample:
    def __init__(self, phrase, y):
        assert(len(phrase) > 0)
        assert(y in [0, 1])
        self.phrase = phrase
        self.y = y


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

    X_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    while True:
        for irow, sample in enumerate(samples):
            vectorize_words(sample.words, X_batch, batch_index, w2v)
            if mode == 1:
                y_batch[batch_index, sample.y] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                xx = {'input_words': X_batch}
                if mode == 1:
                    yield (xx, {'output': y_batch})
                else:
                    yield xx

                # очищаем матрицы порции для новой порции
                X_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0

# -------------------------------------------------------------------


NB_EPOCHS = 1000

# Разбор параметров тренировки в командной строке
parser = argparse.ArgumentParser(description='Neural model for interpretation requirement classifier')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--arch', type=str, default='lstm', help='neural model architecture: lstm | lstm(cnn)')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.bin', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing data files')

args = parser.parse_args()
data_folder = args.data_dir
tmp_folder = args.tmp

wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)
batch_size = args.batch_size
net_arch = args.arch

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_req_interpretation.log'))

run_mode = args.run_mode

config_path = os.path.join(tmp_folder, 'nn_req_interpretation.config')

if run_mode == 'train':
    arch_filepath = os.path.join(tmp_folder, 'nn_req_interpretation.arch')
    weights_path = os.path.join(tmp_folder, 'nn_req_interpretation.weights')
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

    print('Restoring model architecture from {}'.format(arch_filepath))
    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    print('Loading model weights from {}'.format(weights_path))
    model.load_weights(weights_path)

# ------------------------------------------------------------------

tokenizer = Tokenizer()
tokenizer.load()

if run_mode == 'train':
    logging.info('Start with run_mode==train')

    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
    word_dims = embeddings.vector_size

    samples = []  # список из экземпляров Sample

    # И файлов interpretation.txt и interpretation_auto_5.txt возьмем
    # сэмплы с label=1 - которые надо интерпретировать.
    with codecs.open(os.path.join(data_folder, 'interpretation.txt'), 'r', 'utf-8') as rdr:
        phrases = []
        for line in rdr:
            line = line.strip()
            if len(line) == 0:
                if len(phrases) > 0:
                    last = phrases[-1]
                    if '|' in last:
                        px = last.split('|')
                        if px[0].lower() != px[1].lower():
                            samples.append(Sample(px[0], 1))

                phrases = []
            else:
                phrases.append(line)

    # Из автоматического датасета возьмем столько же сэмплов, сколько получилось
    # из ручного датасета.
    samples2 = []
    with codecs.open(os.path.join(data_folder, 'interpretation_auto_5.txt'), 'r', 'utf-8') as rdr:
        phrases = []
        for line in rdr:
            line = line.strip()
            if len(line) == 0:
                if len(phrases) > 0:
                    last = phrases[-1]
                    if '|' in last:
                        px = last.split('|')
                        if px[0].lower() != px[1].lower():
                            samples2.append(Sample(px[0], 1))

                phrases = []
            else:
                phrases.append(line)

    samples2 = np.random.permutation(samples2)
    samples2 = samples2[:len(samples)*2]  # оставим примерно столько автосэмплов, сколько извлечено из ручного датасета
    samples.extend(samples2)

    # Добавляем негативные примеры, то есть такие предложения, для которых
    # не надо выполнять интерпретацию.
    df = pd.read_csv(os.path.join(data_folder, 'premise_question_relevancy.csv'), encoding='utf-8', delimiter='\t', quoting=3)

    for premise in df.premise.values[:len(samples)]:
        samples.append(Sample(premise, 0))

    # Токенизация сэмплов
    tokenizer = Tokenizer()
    for sample in samples:
        sample.words = tokenizer.tokenize(sample.phrase)

    nb_0 = sum(sample.y == 0 for sample in samples)
    nb_1 = sum(sample.y == 1 for sample in samples)
    logging.info('nb_0={} nb_1={}'.format(nb_0, nb_1))

    max_wordseq_len = 0
    for sample in samples:
        max_wordseq_len = max(max_wordseq_len, len(sample.words))

    logging.info('max_wordseq_len={}'.format(max_wordseq_len))

    if padding == 'left':
        for sample in samples:
            sample.words = lpad_wordseq(sample.words, max_wordseq_len)
    else:
        for sample in samples:
            sample.words = rpad_wordseq(sample.words, max_wordseq_len)

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
    }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    logging.info('Constructing the NN model arch={}...'.format(net_arch))

    nb_filters = 128  # 128
    rnn_size = word_dims*2

    input_words = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_words')

    # суммарный размер выходных тензоров в conv1, то есть это сумма размеров векторов
    # для всех слоев в списке conv1, если их смерджить.
    encoder_size = 0
    layers = []
    if net_arch == 'lstm':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                 input_shape=(max_wordseq_len, word_dims),
                                                 return_sequences=False))

        encoder_rnn = words_rnn(input_words)
        encoder_size = rnn_size*2
        layers.append(encoder_rnn)

    if len(layers) == 1:
        classif = layers[0]
    else:
        classif = keras.layers.concatenate(inputs=layers)

    classif = Dense(units=encoder_size//2, activation='relu')(classif)
    classif = Dense(units=2, activation='softmax', name='output')(classif)
    model = Model(inputs=input_words, outputs=classif)
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

        if istep >= nb_validation_steps:
            break

    f1 = sklearn.metrics.f1_score(y_true=y_true2, y_pred=y_pred2)
    logging.info('val f1={}'.format(f1))

# </editor-fold>


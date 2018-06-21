# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import codecs
import itertools
import json
import os
import sys
import argparse
import random
import math
import pandas as pd
import numpy as np

import keras.callbacks
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Input
from keras.layers import Embedding
from keras.layers.core import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Reshape
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

import gensim

from utils.tokenizer import Tokenizer
from utils.segmenter import Segmenter
from utils.padding_utils import pad_wordseq
from utils.padding_utils import PAD_WORD
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup

input_path = '../data/premise_question_relevancy.csv'
arch0_filepath = '../tmp/train_skip_thoughts.arch'
weights0_path = '../tmp/train_skip_thoughts.weights'
config0_path = '../tmp/train_skip_thoughts.config'

batch_size = 200


def vectorize_phrases(phrases, w2v, word_dims, tokenizer, max_wordseq_len):
    nb_samples = len(phrases)
    X_data = np.zeros((nb_samples, max_wordseq_len, word_dims), dtype='float32')
    for iphrase, phrase in enumerate(phrases):
        words = tokenizer.tokenize(phrase)[:max_wordseq_len]
        for iword, word in enumerate(words):
            if word in w2v:
                X_data[iphrase, iword, :] = w2v[word]

    return X_data


print('Restoring model architecture from {}'.format(arch0_filepath))
with open(arch0_filepath, 'r') as f:
    encoder = model_from_json(f.read())

print('Loading model weights from {}'.format(weights0_path))
encoder.load_weights(weights0_path)

with open(config0_path, 'r') as f:
    model_config = json.load(f)
    word2vector_path = model_config['word2vector_path']
    word_dims = model_config['word_dims']
    max_phrase_len = model_config['max_phrase_len']
    encoder_dim = model_config['encoder_dim']

# Грузим ранее подготовленный датасет для тренировки модели
df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

y_data = np.asarray(df['relevance'].values, dtype='float32')

tokenizer = Tokenizer()

# Анализ и векторизация датасета
max_wordseq_len = 0
for phrase in itertools.chain(df['premise'].values, df['question'].values):
    words = tokenizer.tokenize(phrase)
    max_wordseq_len = max(max_wordseq_len, len(words))

print('max_wordseq_len={}'.format(max_wordseq_len))


w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))

premises = df['premise'].values
questions = df['question'].values

print('Encoding premises...')
X_premises = vectorize_phrases(premises, w2v, word_dims, tokenizer, max_phrase_len)
X_premises = encoder.predict(X_premises, verbose=1)
print('X_premises.shape={}'.format(X_premises.shape))

print('Encoding questions...')
X_questions = vectorize_phrases(questions, w2v, word_dims, tokenizer, max_phrase_len)
X_questions = encoder.predict(X_questions, verbose=1)
print('X_questions.shape={}'.format(X_questions.shape))

encoder_dim = X_premises.shape[1]

SEED = 123456
TEST_SHARE = 0.2
X_premises_train, X_premises_val,\
X_questions_train, X_questions_val,\
y_train, y_val = train_test_split(X_premises, X_questions, y_data,
                                  test_size=TEST_SHARE,
                                  random_state=SEED)

# Модель второго уровня обучается supervisedly по парам из датасета релевантности.
input_premises = Input(batch_shape=(batch_size, encoder_dim), dtype='float32', name='input_premises')
input_questions = Input(batch_shape=(batch_size, encoder_dim), dtype='float32', name='input_questions')

if False:
    addition = add([input_premises, input_questions])
    minus_y1 = Lambda(lambda x: -x, output_shape=(encoder_dim,))(input_premises)
    mul = add([input_questions, minus_y1])
    mul = multiply([mul, mul])

    net = keras.layers.concatenate(inputs=[mul, addition])
else:
    net = keras.layers.concatenate(inputs=[input_premises, input_questions])

net = Dense(units=encoder_dim, activation='relu')(net)
net = Dense(units=encoder_dim // 2, activation='relu')(net)
net = Dense(units=encoder_dim // 3, activation='relu')(net)
net = Dense(units=1, activation='sigmoid', name='output')(net)

model = Model(inputs=[input_premises, input_questions], outputs=net)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.summary()

weights_path = '../tmp/nn_relevancy_skip_thoughts.weights'
model_checkpoint = ModelCheckpoint(weights_path, monitor='val_acc',
                                   verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')

print('Start training on {} samples, validating on {} samples'.format(X_questions_train.shape[0], X_questions_val.shape[0]))
model.fit(x={'input_premises': X_premises_train, 'input_questions': X_questions_train},
          y=y_train,
          validation_data=({'input_premises': X_premises_val, 'input_questions': X_questions_val}, y_val),
          batch_size=batch_size,
          epochs=100,
          callbacks=[model_checkpoint, early_stopping],
          verbose=1)

print('Estimating final accuracy...')
model.load_weights(weights_path)

y_pred = model.predict({'input_premises': X_premises_val, 'input_questions': X_questions_val})
y_pred = y_pred.reshape(y_pred.shape[0])

print('y_pred.shape={} dtype={}'.format(y_pred.shape, y_pred.dtype))
print('y_val.shape={} dtype={}'.format(y_val.shape, y_val.dtype))

f1 = sklearn.metrics.f1_score(y_true=y_val, y_pred=y_pred>0.5)
print('val f1={}'.format(f1))



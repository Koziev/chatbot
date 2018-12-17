# -*- coding: utf-8 -*-
"""
Тренировка модели получения векторов предложений.
На основу взята архитектура Skip-Thoughts.
Предложения берем из большого текстового корпуса, в котором текст разбит на абзацы. Предложения
в одном абзаце считаем связанными по смыслу.

Для оценки качества получающейся модели встраивания берем датасет для тренировки nn_synonymy_tripleloss.py
и
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
from utils.padding_utils import rpad_wordseq
from utils.padding_utils import PAD_WORD
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup


qa_path = '../data/qa.txt'
relevancy_path = '../data/premise_question_relevancy.csv'
#corpus_path = r'f:\Corpus\Raw\ru\text_blocks.txt'
corpus_path = os.path.expanduser('~/Corpus/Raw/ru/text_blocks.txt')
max_nb_samples = 5000000
phrase_vector_dim = 512
batch_size = 256
max_inseq_len = 12
NB_EPOCHS = 100
max_gap = 5  # максимальное расстояние между предложениями из raw корпуса
arch = 'gru'
word2vector_path = os.path.expanduser('~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.bin')
data_folder = '../data'

PAD_WORD = u''

arch_filepath = '../tmp/train_skip_thoughts.arch'
weights_path = '../tmp/train_skip_thoughts.weights'
config_path = '../tmp/train_skip_thoughts.config'


def norm(v):
    """
    Так как векторы в модели word2vec могут иметь значения компонентов за пределами -1...+1 (если не были нормированы),
    то для работы нейросетки нужно нормировать вектор.
    :param v: 
    :return: 
    """
    return v/np.linalg.norm(v)


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


def is_good_phrase(phrase, words, w2v):
    nw = len(words)
    if nw < 3 or nw > max_inseq_len:
        return False

    for word in words:
        if word not in w2v:
            return False

    return True


def sanitize_phrase(phrase, tokenizer):
    words = tokenizer.tokenize(phrase)
    return u' '.join(words), words


def lpad_words(words, n):
    l = len(words)
    if l < n:
        return list(itertools.repeat(PAD_WORD, n-l))+words
    else:
        return words


def normalize_qline( line ):
    line = line.replace(u'(+)', u'')
    line = line.replace(u'(-)', u'')
    line = line.replace(u'T:', u'')
    line = line.replace(u'Q:', u'')
    line = line.replace(u'A:', u'')
    line = line.replace(u'\t', u' ')
    line = line.replace('.', ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ').replace('-', ' ')
    line = line.replace('  ', ' ').strip().lower()
    return line


def sent_proba(gap):
    return math.exp(1-gap)


def vectorize_words(words, x, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            x[irow, iword, :] = word2vec[word]


def generate_rows(samples, batch_size, max_phrase_len, w2v, word_dims, mode):
    batch_index = 0
    batch_count = 0

    X_batch = np.zeros((batch_size, max_phrase_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, max_phrase_len, word_dims), dtype=np.float32)

    while True:
        if mode == 1:
            # При обучении сетки каждую эпоху тасуем сэмплы.
            random.shuffle(samples)

        for irow, (words1, words2) in enumerate(samples):
            vectorize_words(words1, X_batch, batch_index, w2v)
            vectorize_words(words2, y_batch, batch_index, w2v)

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                if mode == 1:
                    yield (X_batch, y_batch)
                else:
                    yield X_batch

                # очищаем матрицы порции для новой порции
                X_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0



print(u'Loading w2v model from {}'.format(word2vector_path))
w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
w2v_dims = len(w2v.syn0[0])

segmenter = Segmenter()
tokenizer = Tokenizer()

print('Collecting samples...')
samples = []
all_words = set([PAD_WORD])
max_phrase_len = 0

if False:
    # добавляем пары предпосылка-вопрос из обучающего датасета

    with codecs.open(os.path.join(data_folder, qa_path), "r", "utf-8") as inf:

        loading_state = 'T'

        text = []
        questions = []  # список пар (вопрос, ответ)

        while True:
            line = inf.readline()
            if len(line) == 0:
                break

            line = line.strip()

            if line.startswith(u'T:'):
                if loading_state == 'T':
                    text.append(normalize_qline(line))
                else:
                    # закончился парсинг предыдущего блока из текста (предпосылки),
                    # вопросов и ответов.
                    if len(text) == 1:
                        for premise in text:
                            for question2 in questions:
                                question = question2[0]

                                phrase1, words1 = sanitize_phrase(question, tokenizer)
                                phrase2, words2 = sanitize_phrase(premise, tokenizer)
                                if is_good_phrase(phrase1, words1, w2v) and is_good_phrase(phrase2, words2, w2v):
                                    samples.append((words1, words2))
                                    all_words.update(words1)
                                    all_words.update(words2)
                                    max_phrase_len = max(len(words1), len(words2), max_phrase_len)

                    loading_state = 'T'
                    questions = []
                    text = [normalize_qline(line)]

            elif line.startswith(u'Q:'):
                loading_state = 'Q'
                q = normalize_qline(line)
                a = normalize_qline(inf.readline().strip())
                questions.append((q, a))


    print('{} samples extracted from from {}'.format(len(samples), qa_path))

if True:
    # добавляем пары соседних предложений (с допустимым max_gap) из большого корпуса.
    print(u'Loading samples from {}'.format(corpus_path))
    nb_from_corpus = 0
    with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
        for line in rdr:
            line = line.strip()
            phrases = segmenter.split(line)
            n = len(phrases)
            if n > 2:
                for i1 in range(n):
                    for i2 in range(i1 + 1, min(n, i1+1+max_gap)):
                        gap = abs(i1-i2)
                        if gap <= max_gap:
                            p = sent_proba(gap)
                            if random.random() <= p:
                                phrase1 = phrases[i1]
                                phrase2 = phrases[i2]
                                phrase1, words1 = sanitize_phrase(phrase1, tokenizer)
                                phrase2, words2 = sanitize_phrase(phrase2, tokenizer)
                                if is_good_phrase(phrase1, words1, w2v) and is_good_phrase(phrase2, words2, w2v):
                                    samples.append((words1, words2))
                                    all_words.update(words1)
                                    all_words.update(words2)
                                    max_phrase_len = max(len(words1), len(words2), max_phrase_len)
                                    nb_from_corpus += 1

                                    if (nb_from_corpus%1000) == 0:
                                        print('{}'.format(nb_from_corpus), end='\r')

                            if len(samples) > max_nb_samples:
                                break

                    if len(samples) > max_nb_samples:
                        break

            if len(samples) > max_nb_samples:
                break

    print('\n')

if False:
    # добавляем пары перефразировок
    df = pd.read_csv(relevancy_path, encoding='utf-8', delimiter='\t', quoting=3)
    for i, row in df[df['relevance'] == 1].iterrows():
        premise = row['premise']
        question = row['question']
        if premise != question:
            phrase1, words1 = sanitize_phrase(question, tokenizer)
            phrase2, words2 = sanitize_phrase(premise, tokenizer)
            if is_good_phrase(phrase1, words1, w2v) and is_good_phrase(phrase2, words2, w2v):
                samples.append((words1, words2))
                all_words.update(words1)
                all_words.update(words2)
                max_phrase_len = max(len(words1), len(words2), max_phrase_len)

if len(samples) > max_nb_samples:
    samples = samples[:max_nb_samples]

print('total number of samples={}'.format(len(samples)))

print('max_phrase_len={} words'.format(max_phrase_len))
print('samples.count={}'.format(len(samples)))

word_dims = w2v_dims


print('Building network model {}'.format(arch))
# на входе каждое предложение представляется цепочкой токенов
input_curr_phrase = Input(batch_shape=(batch_size, max_phrase_len, w2v_dims), dtype='float32', name='input_curr_phrase')
encoder_dim = -1

encoder_curr = input_curr_phrase

if arch == 'gru':
    rnn_size = phrase_vector_dim
    rnn_layer = recurrent.GRU(rnn_size, return_sequences=False)
    encoder_curr = rnn_layer(encoder_curr)
    encoder_dim = phrase_vector_dim
elif arch == 'bilstm':
    rnn_size = phrase_vector_dim
    rnn_layer = Bidirectional(recurrent.LSTM(rnn_size//2, return_sequences=False))
    encoder_curr = rnn_layer(encoder_curr)
    encoder_dim = phrase_vector_dim
elif arch == 'lstm(lstm)':
    rnn_size = phrase_vector_dim
    encoder_curr = recurrent.LSTM(rnn_size//4, return_sequences=True)(encoder_curr)
    encoder_curr = recurrent.LSTM(rnn_size, return_sequences=False)(encoder_curr)
    encoder_dim = rnn_size
elif 'lstm(cnn)':
    # рекуррентные слои поверх сверточных
    convs = []
    for kernel_size, nb_filters in [(1, 64), (2, 128), (3, 256)]:
        conv = Conv1D(filters=nb_filters,
                      kernel_size=kernel_size,
                      padding='valid',
                      activation='relu',
                      strides=1)

        pooler = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')
        # pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')
        # pooler = None

        rnn_size = nb_filters
        # поверх сверточных идут рекуррентные слои
        lstm = Bidirectional(recurrent.GRU(rnn_size, return_sequences=False))

        conv_layer = conv(encoder_curr)
        conv_layer = pooler(conv_layer)
        conv_layer = lstm(conv_layer)

        convs.append(conv_layer)

    encoder_curr = keras.layers.concatenate(inputs=convs)
    encoder_curr = Dense(units=phrase_vector_dim, activation='relu')(encoder_curr)
    encoder_dim = phrase_vector_dim

elif arch == 'cnn':
    # простая сверточная архитектура.
    convs = []
    for kernel_size, nb_filters in [(1, 128), (2, 256), (3, 512)]:
        conv = Conv1D(filters=nb_filters,
                      kernel_size=kernel_size,
                      padding='valid',
                      activation='relu',
                      strides=1)

        pooler = GlobalAveragePooling1D()

        conv_layer = conv(encoder_curr)
        conv_layer = pooler(conv_layer)
        convs.append(conv_layer)

    encoder_curr = keras.layers.concatenate(inputs=convs)
    encoder_curr = Dense(units=phrase_vector_dim, activation='relu')(encoder_curr)
    encoder_dim = phrase_vector_dim

nb_predict_layers = 0
predictor_next = Dense(units=encoder_dim, activation='relu')(encoder_curr)
for _ in range(nb_predict_layers):
    predictor_next = Dense(units=encoder_dim, activation='relu')(predictor_next)

decoder1 = recurrent.LSTM(word_dims, return_sequences=True)
decoder2 = Dense(word_dims, activation='tanh')

decoder_next = RepeatVector(max_phrase_len)(predictor_next)
decoder_next = decoder1(decoder_next)
decoder_next = TimeDistributed(decoder2, name='output_next')(decoder_next)

model = Model(inputs=input_curr_phrase, outputs=decoder_next)
model.compile(loss=keras.losses.mean_squared_error, optimizer='nadam')
model.summary()

nb_samples = len(samples)
X_curr_data = np.zeros((nb_samples, max_phrase_len, word_dims), dtype='float32')
y_next_data = np.zeros((nb_samples, max_phrase_len, word_dims), dtype='float32')

for isample, sample in enumerate(samples):
    for iword, word in enumerate(sample[0]):
        if word != PAD_WORD:
            X_curr_data[isample, iword, :] = norm(w2v[word])

    for iword, word in enumerate(sample[1]):
        y_next_data[isample, iword, :] = norm(w2v[word])

SEED = 123456
TEST_SHARE = 0.2
samples_train, samples_val = train_test_split(samples,
                                               test_size=TEST_SHARE,
                                               random_state=SEED)

model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='auto')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

if NB_EPOCHS == 0:
    model.save_weights(weights_path)
else:
    hist = model.fit_generator(generator=generate_rows(samples_train, batch_size, max_phrase_len, w2v, word_dims, 1),
                               steps_per_epoch=len(samples_train) // batch_size,
                               epochs=NB_EPOCHS,
                               verbose=1,
                               callbacks=[model_checkpoint, early_stopping],
                               validation_data=generate_rows(samples_val, batch_size, max_phrase_len, w2v, word_dims, 1),
                               validation_steps=len(samples_val) // batch_size
                               )

    # сделаем валидацию модели на специальной задаче.
    # для этого нам нужен только энкодер предложения
    model.load_weights(weights_path)

# Усекаем модель до кодирующей части
model2 = Model(inputs=input_curr_phrase, outputs=encoder_curr)
#model2.compile(loss=keras.losses.mean_squared_error, optimizer='nadam')
model2.summary()

# Сохраняем кодер для использования в других моделях
with open(arch_filepath, 'w') as f:
    f.write(model2.to_json())

model2.save_weights(weights_path)

# Для удобства сохраним конфигурацию модели
with open(config_path, 'w') as f:
    model_config = {'word2vector_path': word2vector_path,
                    'word_dims': word_dims,
                    'max_phrase_len': max_phrase_len,
                    'encoder_dim': encoder_dim,
                    'batch_size': batch_size
                    }
    json.dump(model_config, f)


# Оценим точность модели по датасету, используемому для тренировки nn_synonymy_tripleloss.py

class Sample3:
    def __init__(self, anchor, positive, negative):
        assert(len(anchor) > 0)
        assert(len(positive) > 0)
        self.anchor = anchor
        self.positive = positive
        self.negative = negative


for dataset_path in ['../data/relevancy_dataset3.csv', '../data/synonymy_dataset3.csv']:
    print('Vaditating on {}...'.format(dataset_path))
    samples3 = []
    phrases = set()
    df = pd.read_csv(dataset_path, encoding='utf-8', delimiter='\t', quoting=3)
    for anchor, positive, negative in itertools.izip(df['anchor'].values, df['positive'].values, df['negative'].values):
        samples3.append(Sample3(anchor, positive, negative))
        phrases.add(anchor)
        phrases.add(positive)
        phrases.add(negative)

    phrases = list(phrases)
    phrase2index = dict((phrase, i) for (i, phrase) in enumerate(phrases))
    nb_phrases = len(phrases)
    nrow = ((nb_phrases // batch_size)+1)*batch_size
    X_data = np.zeros((nrow, max_phrase_len, word_dims), dtype='float32')

    for irow, phrase in enumerate(phrases):
        words = phrase.split()
        for iword, word in enumerate(words[:max_phrase_len]):
            if word in w2v:
                X_data[irow, iword, :] = norm(w2v[word])

    y_pred = model2.predict(X_data, batch_size=batch_size)
    nb_good = 0
    nb_total = 0
    for sample in samples3:
        v_anchor = y_pred[phrase2index[sample.anchor]]
        v_positive = y_pred[phrase2index[sample.positive]]
        v_negative = y_pred[phrase2index[sample.negative]]
        sim1 = v_cosine(v_anchor, v_positive)
        sim2 = v_cosine(v_anchor, v_negative)
        if sim1 > sim2:
            nb_good += 1

        nb_total += 1

    acc = float(nb_good) / float(nb_total)
    print('{} accuracy={}'.format(dataset_path, acc))



# Другая оценка - выбор предпосылки для вопроса.
# Теперь грузим данные для оценочной задачи
eval_data = EvaluationDataset(max_phrase_len, tokenizer, 'right')
eval_data.load(data_folder)

nb_good = 0  # попадание предпосылки в top-1
nb_good5 = 0
nb_good10 = 0
nb_total = 0

for irecord, phrases in eval_data.generate_groups():
    nb_samples = len(phrases)
    nrow = nb_samples*2
    if (nrow % batch_size) != 0:
        nrow = ((nrow // batch_size)+1) * batch_size

    #print('Building X_data with nrow={}'.format(nrow))
    X_data = np.zeros((nrow, max_phrase_len, word_dims), dtype='float32')

    for irow, (premise_words, question_words) in enumerate(phrases):
        for iword, word in enumerate(premise_words[:max_phrase_len]):
            if word in w2v:
                X_data[irow*2, iword, :] = norm(w2v[word])

        for iword, word in enumerate(question_words[:max_phrase_len]):
            if word in w2v:
                X_data[irow*2+1, iword, :] = norm(w2v[word])

    y_pred = model2.predict(X_data, batch_size=batch_size)

    sims = []
    for i in range(nb_samples):
        v1 = y_pred[i*2]
        v2 = y_pred[i*2+1]
        sim = v_cosine(v1, v2)
        sims.append(sim)

    # предпосылка с максимальной релевантностью
    max_index = np.argmax(sims)
    selected_premise = phrases[max_index][0]

    nb_total += 1

    # эта выбранная предпосылка соответствует одному из вариантов
    # релевантных предпосылок в этой группе?
    if eval_data.is_relevant_premise(irecord, selected_premise):
        nb_good += 1
        nb_good5 += 1
        nb_good10 += 1
        #print(EvaluationMarkup.ok_color + EvaluationMarkup.ok_bullet + EvaluationMarkup.close_color, end='')
    else:
        #print(EvaluationMarkup.fail_color + EvaluationMarkup.fail_bullet + EvaluationMarkup.close_color, end='')

        # среди top-5 или top-10 предпосылок есть верная?
        sorted_phrases = [x for x, _ in sorted(itertools.izip(phrases, sims), key=lambda z: -z[1])]

        for i in range(1, 10):
            selected_premise = u' '.join(sorted_phrases[i][0]).strip()
            if eval_data.is_relevant_premise(irecord, selected_premise):
                if i < 5:
                    nb_good5 += 1  # верная предпосылка вошла в top-5
                if i < 10:
                    nb_good10 += 1
                break

    max_sim = np.max(y_pred)

    question = phrases[0][1]
    #print(u'{:<40} {:<40} {}/{}'.format(question, phrases[max_index][0], sims[max_index], sims[0]))


# Итоговая точность выбора предпосылки.
accuracy = float(nb_good) / float(nb_total)
print('accuracy       ={}'.format(accuracy))

# Также выведем точность попадания верной предпосылки в top-5 и top-10
accuracy5 = float(nb_good5) / float(nb_total)
print('accuracy top-5 ={}'.format(accuracy5))

accuracy10 = float(nb_good10) / float(nb_total)
print('accuracy top-10={}'.format(accuracy10))

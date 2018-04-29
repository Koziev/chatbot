# -*- coding: utf-8 -*-
"""
Модель для определения релевантности предпосылки и вопроса.
Модель используется в проекте чат-бота https://github.com/Koziev/chatbot

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

import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.merge import concatenate, add, multiply
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer
from utils.segmenter import Segmenter


# Включать ли сэмплы, в которых задается релевантность предпосылки и вопроса,
# например:
# premise=Кошка ловит мышей
# question=Кто ловит мышей?
INCLUDE_PREMISE_QUESTION = False



# -------------------------------------------------------------------

PAD_WORD = u''


def pad_wordseq(words, n):
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words,))

# -------------------------------------------------------------------


# Разбор параметров тренировки, указанных в командной строке
parser = argparse.ArgumentParser(description='Neural model for text relevance estimation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train evaluate query query2')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture, one of lstm lstm(cnn) (lstm)cnn cnn')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--max_nb_samples', type=int, default=1000000000, help='upper limit for number of samples')
parser.add_argument('--input', type=str, default='../data/premise_question_relevancy.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=48.txt', help='path to word2vector model dataset')

args = parser.parse_args()

input_path = args.input
tmp_folder = args.tmp
wordchar2vector_path = args.wordchar2vector
word2vector_path = args.word2vector
batch_size = args.batch_size
net_arch = args.arch

# Для быстрого проведения исследований влияния гиперпараметров удобно брать для
# обучения не полный датасет, а небольшую часть - указываем кол-во паттернов.
max_nb_samples = args.max_nb_samples

run_mode = args.run_mode
# Варианты значения для параметра run_mode:
# run_mode='train'
# Тренировка модели по заранее приготовленному датасету с парами предложений

# run_mode = 'evaluate'

# run_mode = 'query'
# В консоли вводятся два предложения, модель оценивает их релевантность.

# run_mode = 'query2'
# В консоли вводится имя текстового файла со списком предложений
# в второе проверяемое предложение, модель выводит список предложений,
# отсортированный по релевантности.


max_wordseq_len = 0
max_charseq_len = 0
all_words = set()
all_chars = set()

# --------------------------------------------------------------------------

# Грузим заранее подготовленные векторы слов для модели
# встраивания wordchar2vector (см. wordchar2vector.py)
print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])
print('wc2v_dims={0}'.format(wc2v_dims))

# --------------------------------------------------------------------------

# Грузим ранее подготовленный датасет для тренировки модели
df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

# Анализ и векторизация датасета
tokenizer = Tokenizer()
for phrase in itertools.chain( df['premise'].values, df['question'].values ):
    all_chars.update( phrase )
    max_charseq_len = max( max_charseq_len, len(phrase) )
    words = tokenizer.tokenize(phrase)
    all_words.update(words)
    max_wordseq_len = max( max_wordseq_len, len(words) )

for word in wc2v.vocab:
    all_words.add(word)
    all_chars.update(word)

print('max_charseq_len={}'.format(max_charseq_len))
print('max_wordseq_len={}'.format(max_wordseq_len))

nb_chars = len(all_chars)
nb_words = len(all_words)
print('nb_chars={}'.format(nb_chars))
print('nb_words={}'.format(nb_words))

# --------------------------------------------------------------------------

print( 'Loading the w2v model {}'.format(word2vector_path) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=False)
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

del w2v
del wc2v
gc.collect()
# -------------------------------------------------------------------


print('Constructing the NN model...')

nb_filters = 150  # 128
rnn_size = word_dims

final_merge_size = 0

arch_filepath = os.path.join(tmp_folder, 'relevancy_model.arch')
weights_path = os.path.join(tmp_folder, 'relevancy.weights')

# сохраним конфиг модели, чтобы ее использовать в чат-боте
model_config = {
                'model': 'nn',
                'max_wordseq_len': max_wordseq_len,
                'w2v_path': word2vector_path,
                'wordchar2vector_path': wordchar2vector_path,
                'PAD_WORD': PAD_WORD,
                'arch_filepath': arch_filepath,
                'weights_path': weights_path,
                'word_dims': word_dims
               }

with open(os.path.join(tmp_folder,'relevancy_model.config'), 'w') as f:
    json.dump(model_config, f)

# ------------------------------------------------------------------

if run_mode == 'train':
    if net_arch == 'lstm':
        words_net1 = Input(shape=(max_wordseq_len,word_dims,), dtype='float32', name='input_words1')
        words_net2 = Input(shape=(max_wordseq_len,word_dims,), dtype='float32', name='input_words2')

        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_wordseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        if False:
            # первый вариант вычисления похожести
            words_merged = keras.layers.multiply(inputs=[encoder_rnn1, encoder_rnn2])
            words_final = Dense(units=int(rnn_size*2), activation='relu')(words_merged)
            words_final = Dense(units=int(rnn_size/2), activation='relu')(words_final)
            #words_final = Dense(units=int(rnn_size/4), activation='relu')(words_final)
        else:
            # второй вариант вычисления похожести
            addition = add([encoder_rnn1, encoder_rnn2])
            minus_y1 = Lambda(lambda x: -x, output_shape=(rnn_size * 2,))(encoder_rnn1)
            mul = add([encoder_rnn2, minus_y1])
            mul = multiply([mul, mul])
            muladd = concatenate([mul, addition])
            words_final = keras.layers.concatenate(inputs=[encoder_rnn1, muladd, encoder_rnn2])
            words_final = Dense(units=int(rnn_size*2), activation='relu')(words_final)
            words_final = Dense(units=int(rnn_size/2), activation='relu')(words_final)

    # --------------------------------------------------------------------------

    if net_arch == '(lstm)cnn':

        words_net1 = Input(shape=(max_wordseq_len,word_dims,), dtype='float32', name='input_words1')
        words_net2 = Input(shape=(max_wordseq_len,word_dims,), dtype='float32', name='input_words2')

        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_wordseq_len, word_dims),
                                                        return_sequences=True))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        conv_list1 = []
        conv_list2 = []
        merged_size = 0
        for kernel_size in range(2, 5):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            conv_layer1 = conv(encoder_rnn1)
            conv_layer1 = GlobalMaxPooling1D()(conv_layer1)
            conv_list1.append(conv_layer1)

            conv_layer2 = conv(encoder_rnn2)
            conv_layer2 = GlobalMaxPooling1D()(conv_layer2)
            conv_list2.append(conv_layer2)

            merged_size += nb_filters

        encoder_rnn1 = keras.layers.concatenate(inputs=conv_list1)
        encoder_rnn2 = keras.layers.concatenate(inputs=conv_list2)

        addition = add([encoder_rnn1, encoder_rnn2])
        minus_y1 = Lambda(lambda x: -x, output_shape=(merged_size,))(encoder_rnn1)
        mul = add([encoder_rnn2, minus_y1])
        mul = multiply([mul, mul])
        muladd = concatenate([mul, addition])
        words_final = keras.layers.concatenate(inputs=[encoder_rnn1, muladd, encoder_rnn2])
        words_final = Dense(units=int(rnn_size * 2), activation='relu')(words_final)
        words_final = Dense(units=int(rnn_size / 2), activation='relu')(words_final)

    # --------------------------------------------------------------------------

    if net_arch == 'lstm+cnn':
        words_net1 = Input(shape=(max_wordseq_len,word_dims,), dtype='float32', name='input_words1')
        words_net2 = Input(shape=(max_wordseq_len,word_dims,), dtype='float32', name='input_words2')

        conv1 = []
        conv2 = []

        repr_size = 0

        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_wordseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        dense1 = Dense(units=rnn_size*2)

        #encoder_rnn1 = dense1(encoder_rnn1)
        #encoder_rnn2 = dense1(encoder_rnn2)

        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)

        repr_size += rnn_size*2

        # добавляем входы со сверточными слоями
        for kernel_size in range(2, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            dense2 = Dense(units=nb_filters)

            conv_layer1 = conv(words_net1)
            conv_layer1 = GlobalMaxPooling1D()(conv_layer1)
            #conv_layer1 = dense2(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = GlobalMaxPooling1D()(conv_layer2)
            #conv_layer2 = dense2(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += nb_filters

        encoder_rnn1 = keras.layers.concatenate(inputs=conv1)
        encoder_rnn2 = keras.layers.concatenate(inputs=conv2)

        addition = add([encoder_rnn1, encoder_rnn2])
        minus_y1 = Lambda(lambda x: -x, output_shape=(repr_size,))(encoder_rnn1)
        mul = add([encoder_rnn2, minus_y1])
        mul = multiply([mul, mul])
        muladd = concatenate([mul, addition])
        words_final = keras.layers.concatenate(inputs=[encoder_rnn1, muladd, encoder_rnn2])
        words_final = Dense(units=int(repr_size), activation='relu')(words_final)
        words_final = Dense(units=int(rnn_size / 2), activation='relu')(words_final)

    # --------------------------------------------------------------------------

    if net_arch == 'lstm(cnn)':

        words_net1 = Input(shape=(max_wordseq_len,word_dims,), dtype='float32', name='input_words1')
        words_net2 = Input(shape=(max_wordseq_len,word_dims,), dtype='float32', name='input_words2')

        conv1 = []
        conv2 = []
        encoder_size = 0

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

            conv_layer1 = conv(words_net1)
            conv_layer1 = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer2)
            conv_layer2 = lstm(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += rnn_size

        encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2)))
        words_final = Dense(units=int(encoder_size), activation='relu')(encoder_merged)
        #words_final = BatchNormalization()(words_final)
        words_final = Dense(units=int(encoder_size), activation='relu')(words_final)
        #words_final = BatchNormalization()(words_final)
        #words_final = Dense(units=int(encoder_size / 2), activation='relu')(words_final)
        #words_final = BatchNormalization()(words_final)

    # Вычислительный граф сформирован, добавляем финальный классификатор с 1 выходом
    classif = Dense(units=1, activation='sigmoid', name='output')(words_final)

    model = Model(inputs=[words_net1, words_net2], outputs=classif)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())


else:
    # загружаем ранее натренированную сетку
    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

# -------------------------------------------------------------------------


phrases = []
ys = []

for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
    phrase1 = row['premise']
    phrase2 = row['question']
    words1 = pad_wordseq(tokenizer.tokenize(phrase1), max_wordseq_len)
    words2 = pad_wordseq(tokenizer.tokenize(phrase2), max_wordseq_len)

    y = row['relevance']
    if INCLUDE_PREMISE_QUESTION:
        y = y > 0
        ys.append(int(y))
        phrases.append((words1, words2, phrase1, phrase2))
    elif y in (0, 1):
        ys.append(y)
        phrases.append((words1, words2, phrase1, phrase2))

if len(phrases) > max_nb_samples:
    print('Reducing the list of samples from {} to {} items'.format(len(phrases), max_nb_samples))
    #iphrases = list(np.random.permutation(range(len(phrases))))
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


# -----------------------------------------------------------------


def vectorize_words(words, X_batch, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            X_batch[irow, iword, :] = word2vec[word]


def generate_rows(sequences, targets, batch_size, mode):
    batch_index = 0
    batch_count = 0

    X1_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size), dtype=np.bool)

    while True:
        for irow, (seq,target) in enumerate(itertools.izip(sequences,targets)):
            vectorize_words(seq[0], X1_batch, batch_index, word2vec)
            vectorize_words(seq[1], X2_batch, batch_index, word2vec)
            y_batch[batch_index] = target

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                # print('mode={} batch_count={}'.format(mode, batch_count))
                if mode == 1:
                    yield ({'input_words1': X1_batch, 'input_words2': X2_batch}, {'output': y_batch})
                else:
                    yield {'input_words1': X1_batch, 'input_words2': X2_batch}

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0

# ---------------------------------------------------------------

# <editor-fold desc="train">
if run_mode == 'train':
    print('Start training...')
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_acc',
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')

    hist = model.fit_generator(generator=generate_rows(train_phrases, train_ys, batch_size, 1),
                               steps_per_epoch=int(len(train_phrases)/batch_size),
                               epochs=200,
                               verbose=1,
                               callbacks=[model_checkpoint, early_stopping],
                               validation_data=generate_rows( val_phrases, val_ys, batch_size, 1),
                               validation_steps=int(len(val_phrases)/batch_size),
                               )
    max_acc = max(hist.history['val_acc'])
    print('max val_acc={}'.format(max_acc))

# </editor-fold>

# <editor-fold desc="evaluate">
if run_mode == 'evaluate':
    # Оценка качества натренированной модели на полном тренировочном датасете,
    # сохранение для каждой пары предложений полученного и эталонного результата.
    print('Evaluate {} patterns...'.format(len(phrases)))
    y_pred = model.predict_generator(generator=generate_rows( phrases, ys, batch_size, 2),
                                     steps=int(len(phrases)/batch_size), verbose=1)
    with codecs.open('../tmp/evaluation.txt', 'w', 'utf-8') as wrt:
        for i in range(y_pred.shape[0]):
            phrase1 = u' '.join(phrases[i][0]).strip()
            phrase2 = u' '.join(phrases[i][1]).strip()
            wrt.write(u'{}\n{}\ny_true={} y_pred={}\n\n'.format(phrase1, phrase2, ys[i], y_pred[i]))
# </editor-fold>

# <editor-fold desc="query">
if run_mode == 'query':
    # Ввод двух предложений с клавиатуры и выдача их релевантности.
    X1_probe = np.zeros((1, max_wordseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((1, max_wordseq_len, word_dims), dtype=np.float32)

    while True:
        print('\nEnter two phrases:')
        phrase1 = raw_input('phrase #1:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase) == 0:
            break

        phrase2 = raw_input('phrase #2:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase) == 0:
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
            sim = y_probe[0]
            print('sim={}'.format(sim))

            if False:
                # содержимое X*_probe для отладки
                with open('../tmp/X_probe.rnn_detector.txt', 'w') as wrt:
                    for X, name in [(X1_probe,'X1_probe'), (X2_probe,'X2_probe')]:
                        wrt.write('{}\n'.format(name))
                        for i in range(X.shape[1]):
                            for j in range(X.shape[2]):
                                wrt.write(' {}'.format(X[0,i,j]))
                            wrt.write('\n')
                    wrt.write('\n\n')
# </editor-fold>

# <editor-fold desc="query2">
if run_mode == 'query2':
    # С клавиатуры задается путь к файлу с предложениями, и второе предложение.
    # Модель делает оценку релевантности для каждого предложения в файле и введенного предложения,
    # и сохраняет список оценок с сортировкой.
    path1 = raw_input('path to text file with phrases:\n> ').decode(sys.stdout.encoding).strip().lower()

    segmenter = Segmenter()
    phrases0 = segmenter.split(codecs.open(path1, 'r', 'utf-8').readlines())
    phrases1 = []
    for phrase in enumerate(phrases):
        words = tokenizer.tokenize(phrase)
        if len(words)>0:
            if len(words) >= max_wordseq_len:
                words = words[:max_wordseq_len]
            phrases1.append(words)

    nb_phrases = len(phrases1)
    print(u'File {} contains {} sentences'.format(path1, nb_phrases))

    X1_probe = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)

    for iphrase, words1 in enumerate(phrases1):
        vectorize_words(pad_wordseq(words1, max_wordseq_len), X1_probe, iphrase, word2vec)

    while True:
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
            for i in range(nb_phrases):
                vectorize_words(pad_wordseq(words2, max_wordseq_len), X2_probe, i, word2vec)

            y_probe = model.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})

            sent_rels = [(phrases1[i], y_probe[i]) for i in range(nb_phrases)]
            sent_rels = sorted(sent_rels, key=lambda z:-z[1])

            # Выведем top N фраз из файла
            for phrase1, rel in sent_rels[0:10]:
                print(u'{}\t{}'.format(rel, phrase1))

            print('\n\n')

# </editor-fold>

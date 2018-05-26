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
import sys
import gensim
import numpy as np
import pandas as pd
import tqdm
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

from utils.tokenizer import Tokenizer

input_path = '../data/premise_question_answer.csv'
tmp_folder = '../tmp'
data_folder = '../data'

# Вариант архитектуры нейросетки:
# lstm - один общий bidir LSTM слой упаковывает слова предпосылки и вопроса в векторы. Затем два
#        получившихся вектора поступают на афинные слои классификатора.
# lstm+cnn - параллельно с LSTM слоями из первого варианта работают сверточные слои. Затем
#        все векторы соединяются и поступают на классификатор.
# lstm(cnn) - сначала группа сверточных слоев выделяет словосочетания, затем после пулинга
#        эти признаки поступают на LSTM и далее на классификатор.
#NET_ARCH = 'lstm'
#NET_ARCH = 'lstm+cnn'
NET_ARCH = 'lstm(cnn)'

# Размер минибатчей.
# Менять следует с осторожностью, так как по результатам экспериментов
# уменьшение этого параметра модель обучается медленнее, но становится точнее.
BATCH_SIZE = 200

# -------------------------------------------------------------------

PAD_WORD = u''

# -------------------------------------------------------------------

RUN_MODE = ''
TRAIN_MODEL = ''

while True:
    print('t - train')
    print('q - query')
    a1 = raw_input(':> ')

    if a1 == 't':
        RUN_MODE = 'train'
        TRAIN_MODEL = 'answer_length'
        break
    elif a1 == 'q':
        RUN_MODE = 'query'
        break
    else:
        print('Unrecognized choice "{}"'.format(a1))


max_inputseq_len = 0
max_outputseq_len = 0 # максимальная длина ответа
all_words = set()
all_chars = set()

# --------------------------------------------------------------------------

wordchar2vector_path = os.path.join(tmp_folder,'wordchar2vector.dat')
print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])
print('wc2v_dims={0}'.format(wc2v_dims))

# --------------------------------------------------------------------------

df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

print('samples.count={}'.format(df.shape[0]))

answerlen2count = Counter()
tokenizer = Tokenizer()
for i,record in df.iterrows():
    for phrase in [record['premise'], record['question']]:
        all_chars.update( phrase )
        words = tokenizer.tokenize(phrase)
        all_words.update(words)
        max_inputseq_len = max( max_inputseq_len, len(words) )

    phrase = record['answer']
    all_chars.update(phrase)
    words = tokenizer.tokenize(phrase)
    all_words.update(words)
    max_outputseq_len = max(max_outputseq_len, len(words))

    answer = record['answer'].lower().strip()
    words = tokenizer.tokenize(answer)
    answerlen2count[len(words)] += 1

for word in wc2v.vocab:
    all_words.add(word)
    all_chars.update(word)

print('max_inputseq_len={}'.format(max_inputseq_len))
print('max_outputseq_len={}'.format(max_outputseq_len))

word2id = dict([(c,i) for i,c in enumerate(itertools.chain([PAD_WORD], filter(lambda z:z!=PAD_WORD,all_words)))])

nb_chars = len(all_chars)
nb_words = len(all_words)
print('nb_chars={}'.format(nb_chars))
print('nb_words={}'.format(nb_words))

print('Count of answers by number of words:')
for l,n in sorted(answerlen2count.iteritems(), key=lambda z:z[0]):
    print('number of words={} count={}'.format(l, n))


# --------------------------------------------------------------------------

#w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=32.txt'
w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=48.txt'
#w2v_path = '/home/eek/polygon/WordSDR2/sdr.dat'
#w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=128.txt'
#w2v_path = r'f:\Word2Vec\word_vectors_cbow=1_win=5_dim=32.txt'
print( 'Loading the w2v model {}'.format(w2v_path) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
w2v_dims = len(w2v.syn0[0])
print('w2v_dims={0}'.format(w2v_dims))

word_dims = w2v_dims+wc2v_dims


word2vec = dict()
for word in wc2v.vocab:
    v = np.zeros( word_dims )
    v[w2v_dims:] = wc2v[word]
    if word in w2v:
        v[:w2v_dims] = w2v[word]

    word2vec[word] = v

del w2v
#del wc2v
gc.collect()
# -------------------------------------------------------------------

print('Constructing the NN model...')

nb_filters = 128
rnn_size = word_dims

final_merge_size = 0

# --------------------------------------------------------------------------------

# сохраним конфиг модели, чтобы ее использовать в чат-боте
model_config = {
                'max_inputseq_len': max_inputseq_len,
                'max_outputseq_len': max_outputseq_len,
                'w2v_path': w2v_path,
                'wordchar2vector_path': wordchar2vector_path,
                'PAD_WORD': PAD_WORD,
                'model_folder': tmp_folder,
                'word_dims': word_dims
               }

with open(os.path.join(tmp_folder,'qa_model.config'), 'w') as f:
    json.dump(model_config, f)

# ------------------------------------------------------------------

# В этих файлах будем сохранять натренированную сетку
arch_filepath = os.path.join(tmp_folder, 'qa_{}.arch'.format(TRAIN_MODEL))
weights_path = os.path.join(tmp_folder, 'qa_{}.weights'.format(TRAIN_MODEL))

# ------------------------------------------------------------------

padding = 'left'

if RUN_MODE == 'train':

    print('Building the NN computational graph...')

    words_net1 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words1')
    words_net2 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words2')

    if NET_ARCH == 'lstm':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        encoder_merged = keras.layers.concatenate(inputs=[encoder_rnn1, encoder_rnn2])
        encoder_size = rnn_size*2
        encoder_final = Dense(units=int(encoder_size), activation='softmax')(encoder_merged)

    # --------------------------------------------------------------------------

    if NET_ARCH == 'lstm+cnn':
        conv1 = []
        conv2 = []
        conv3 = []

        repr_size = 0

        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        dense1 = Dense(units=rnn_size*2)
        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)

        repr_size += rnn_size*2

        # добавляем входы со сверточными слоями
        for kernel_size in range(1, 4):
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

            repr_size += nb_filters

        encoder_size = repr_size
        encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2, conv3)))
        encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_merged)
        encoder_size = repr_size

    # --------------------------------------------------------------------------

    if NET_ARCH == 'lstm(cnn)':

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
        encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_merged)


    # финальный классификатор определяет длину ответа
    output_dims = max_outputseq_len
    decoder = Dense(rnn_size, activation='relu')(encoder_final)
    decoder = Dense(int(rnn_size/2), activation='relu')(decoder)
    decoder = Dense(output_dims, activation='softmax', name='output')(decoder)

    model = Model(inputs=[words_net1, words_net2], outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    keras.utils.plot_model(model,
                           to_file=os.path.join(tmp_folder, 'nn_answer_length-{}.arch.png'.format(NET_ARCH)),
                           show_shapes=False,
                           show_layer_names=True)


# -------------------------------------------------------------------------


def pad_wordseq(words, n):
    """ Слева добавляем пустые слова, чтобы длина строки words стала равна n """
    return list( itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words ) )


def rpad_wordseq(words, n):
    """ Справа добавляем пустые слова, чтобы длина строки words стала равна n """
    return list( itertools.chain( words, itertools.repeat(PAD_WORD, n-len(words)) ) )

# -------------------------------------------------------------------------

input_data = []
output_data = []

for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
    premise = row['premise']
    question = row['question']
    answer = row['answer']

    premise_words = pad_wordseq( tokenizer.tokenize(premise), max_inputseq_len )
    question_words = pad_wordseq( tokenizer.tokenize(question), max_inputseq_len )
    #premise_words = rpad_wordseq( tokenizer.tokenize(premise), max_inputseq_len )
    #question_words = rpad_wordseq( tokenizer.tokenize(question), max_inputseq_len )

    answer_words = tokenizer.tokenize(answer)
    input_data.append( (premise_words, question_words, premise, question) )
    output_data.append( (answer_words, answer) )

SEED = 123456
TEST_SHARE = 0.2
train_input, val_input, train_output, val_output = train_test_split( input_data,
                                                                     output_data,
                                                                     test_size=TEST_SHARE,
                                                                     random_state=SEED )

# -----------------------------------------------------------------


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate( words ):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


# -------------------------------------------------------------------

def select_patterns( sequences, targets ):
    sequences1 = []
    targets1 = []

    for seq, target in itertools.izip(sequences, targets):
        sequences1.append(seq)
        targets1.append(target)

    return (sequences1, targets1)


# --------------------------------------------------------------------------------------


def generate_rows( sequences, targets, batch_size, mode ):
    batch_index = 0
    batch_count = 0

    X1_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, output_dims), dtype=np.bool)

    weights = np.zeros((batch_size))
    weights.fill(1.0)

    while True:
        for irow, (seq,target) in enumerate(itertools.izip(sequences,targets)):
            vectorize_words(seq[0], X1_batch, batch_index, word2vec )
            vectorize_words(seq[1], X2_batch, batch_index, word2vec )

            y_batch[batch_index, len(target[0])-1] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield ({'input_words1': X1_batch, 'input_words2': X2_batch}, {'output': y_batch}, weights)
                else:
                    yield {'input_words1': X1_batch, 'input_words2': X2_batch}

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


# -----------------------------------------------------------------

def count_words(words):
    """ Возвращает кол-во непустых строк в последовательности words """
    return len(filter(lambda z:z!=PAD_WORD,words))


batch_size = BATCH_SIZE

if RUN_MODE == 'train':

    nb_train_patterns = len(train_input)
    nb_valid_patterns = len(val_input)

    print('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_acc'

    model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.00001)

    callbacks = [reduce_lr, model_checkpoint, early_stopping]

    hist = model.fit_generator(generator=generate_rows(train_input, train_output, batch_size, 1),
                               steps_per_epoch=int(nb_train_patterns/batch_size),
                               epochs=200,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows( val_input, val_output, batch_size, 1),
                               validation_steps=int(nb_valid_patterns/batch_size)
                               )


if RUN_MODE == 'query':

    padding = 'left'

    arch_filepath = os.path.join(tmp_folder, 'qa_answer_length.arch')
    weights_path = os.path.join(tmp_folder, 'qa_answer_length.weights')

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    X1_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)

    while True:
        print('\nEnter two phrases:')
        phrase1 = raw_input('premise :> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase1)==0:
            break

        phrase2 = raw_input('question:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase2)==0:
            break

        words1 = tokenizer.tokenize(phrase1)
        words2 = tokenizer.tokenize(phrase2)

        words1 = pad_wordseq(words1, max_inputseq_len)
        words2 = pad_wordseq(words2, max_inputseq_len)
        #words1 = rpad_wordseq(words1, max_inputseq_len)
        #words2 = rpad_wordseq(words2, max_inputseq_len)

        X1_probe.fill(0)
        X2_probe.fill(0)

        vectorize_words(words1, X1_probe, 0, word2vec )
        vectorize_words(words2, X2_probe, 0, word2vec )
        y_probe = model.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})

        print('{}'.format(np.argmax(y_probe[0])+1))

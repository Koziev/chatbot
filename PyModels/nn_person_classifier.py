# -*- coding: utf-8 -*-
"""
Тренер нейросетевой модели для определения грамматического лица фразы.
Модель используется в проекте вопросно-ответной системы https://github.com/Koziev/chatbot

Для обучения требуются ранее сгенерированные датасеты:
change_person_1s_2s_dataset_4.csv
change_person_1s_2s_dataset_5.csv
...
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import codecs
import gc
import itertools
import json
import os
import sys

import gensim
import keras.callbacks
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer

# Размер минибатчей при обучении сетки.
# Сильно влияет на достигаемую точность!
batch_size = 50

# Архитектура нейросетки
# cnn - чисто сверточная
# lstm+cnn гибридная сеть, рекуррентные и сверточные слои работают параллельно
# cnn*lstm - рекуррентный слой работает поверх сверточных слоев
NET_ARCH = 'cnn*lstm'

input_paths_1s = [ '../data/change_person_1s_2s_dataset_4.csv',
                   '../data/change_person_1s_2s_dataset_5.csv' ]

# отсюда возьмем список вопросов с подлежащем в 3м лице
qa_paths_3 = ['../data/premise_question_answer6.txt',
              '../data/premise_question_answer5.txt',
              '../data/premise_question_answer4.txt',
              '../data/premise_question_answer_neg4.txt',
              '../data/premise_question_answer_neg5.txt',
              '../data/premise_question_answer_names4.txt'
              ]

# отсюда возьмем список вопросов с подлежащем в 1м лице
qa_paths_1s = ['../data/premise_question_answer4_1s.txt',
               '../data/premise_question_answer5_1s.txt',
               '../data/premise_question_answer_neg4_1s.txt',
               '../data/premise_question_answer_neg5_1s.txt',
               '../data/premise_question_answer_names4_1s.txt'
               ]

# отсюда возьмем список вопросов с подлежащем во 2м лице
qa_paths_2s = ['../data/premise_question_answer4_2s.txt',
               '../data/premise_question_answer5_2s.txt',
               '../data/premise_question_answer_neg4_2s.txt',
               '../data/premise_question_answer_neg5_2s.txt',
               '../data/premise_question_answer_names4_2s.txt'
               ]

tmp_folder = '../tmp'
data_folder = '../data'

# -------------------------------------------------------------------

PAD_WORD = u''
SEED = 123456
TEST_SHARE = 0.2

# -------------------------------------------------------------------


# Слева добавляем пустые слова
def pad_wordseq(words, n):
    return list( itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words ) )


# Справа добавляем пустые слова
def rpad_wordseq(words, n):
    return list( itertools.chain( words, itertools.repeat(PAD_WORD, n-len(words)) ) )


def vectorize_words( words, M, irow, word2vec ):
    for iword,word in enumerate( words ):
        if word!=PAD_WORD and word in word2vec:
            M[irow, iword, :] = word2vec[word]


def generate_rows_classifier( sequences, targets, batch_size, mode ):
    batch_index = 0
    batch_count = 0
    y_encoder = {'1s':0, '2s':1, '3':2}

    X_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, output_dims), dtype=np.bool)

    while True:
        for irow, (seq,target) in enumerate(itertools.izip(sequences,targets)):
            vectorize_words(seq, X_batch, batch_index, word2vec )
            y_batch[batch_index, y_encoder[target]] = True
            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                if mode == 1:
                    yield ({'input_words': X_batch}, {'output': y_batch})

                # очищаем матрицы порции для новой порции
                X_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


# -------------------------------------------------------------------

config_path = os.path.join(tmp_folder, 'person_classifier.config')


RUN_MODE = ''

while True:
    print('t - train classifier')
    print('q - run queries')
    a1 = raw_input(':> ')

    if a1=='t':
        RUN_MODE = 'train'
        break
    elif a1 == 'q':
        RUN_MODE = 'query'
        break
    else:
        print('Unrecognized choice "{}"'.format(a1))


max_inputseq_len = 0

# --------------------------------------------------------------------------

all_words = set([PAD_WORD])

wordchar2vector_path = os.path.join(data_folder,'wordchar2vector.dat')
print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])
print('wc2v_dims={0}'.format(wc2v_dims))

for word in wc2v.vocab:
    all_words.add(word)


word2id = dict( [(c,i) for i,c in enumerate( itertools.chain([PAD_WORD], filter(lambda z:z!=PAD_WORD, all_words)))] )

nb_words = len(all_words)
print('nb_words={}'.format(nb_words))


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


def all_words_known(word2vec, words):
    for word in words:
        if word not in word2vec:
            return False

    return True


tokenizer = Tokenizer()

if RUN_MODE=='train':
    # для тренировки классификатора нам надо взять все предпосылки из датасетов
    # по группам 1е, 2е, 3е лицо.
    input_phrases = []
    classif_ys = []

    for (ds_paths, ds_person) in [(qa_paths_1s, '1s'), (qa_paths_2s, '2s'), (qa_paths_3, '3')]:
        for ds_path in ds_paths:
            print(u'Processing {}'.format(ds_path))
            with codecs.open(ds_path, 'r', 'utf-8') as rdr:
                for line in rdr:
                    if line.startswith(u'T:'):
                        premise = line.replace(u'T:', u'').strip()
                        input_phrases.append(premise)
                        classif_ys.append(ds_person)
                    elif line.startswith(u'Q:'):
                        premise = line.replace(u'Q:', u'').strip()
                        input_phrases.append(premise)
                        classif_ys.append(ds_person)

    # также добавим фраз из датасетов для тренировки менятеля лица
    for ds_path in input_paths_1s:
        print(u'Processing {}'.format(ds_path))
        with codecs.open(ds_path, 'r', 'utf-8') as rdr:
            for line in rdr:
                cols = line.strip().split(u'\t')
                phrase_1s = cols[0]
                phrase_2s = cols[1]
                input_phrases.append(phrase_1s)
                classif_ys.append('1s')
                input_phrases.append(phrase_2s)
                classif_ys.append('2s')

    print('Total number of patterns={}'.format(len(input_phrases)))

    n_1s = len(filter(lambda z:z=='1s', classif_ys))
    n_2s = len(filter(lambda z:z=='2s', classif_ys))
    n_3 = len(filter(lambda z:z=='3', classif_ys))
    print('Classes:\n1s => {}\n2s => {}\n3  => {}'.format(n_1s, n_2s, n_3))


    input_words = []
    for phrase in input_phrases:
        words = tokenizer.tokenize(phrase)
        all_words.update(words)
        max_inputseq_len = max(max_inputseq_len, len(words))
        input_words.append(words)

    input_words = [ pad_wordseq(wx, max_inputseq_len) for wx in input_words ]

    print('max_inputseq_len={} words'.format(max_inputseq_len))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'engine': 'nn',
        'max_inputseq_len': max_inputseq_len,
        'w2v_path': w2v_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'model_folder': tmp_folder,
        'word_dims': word_dims
    }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    # В этих файлах будем сохранять натренированную сетку
    arch_filepath = os.path.join(tmp_folder, 'person_classifier.arch')
    weights_path = os.path.join(tmp_folder, 'person_classifier.weights')

    # --------------------------------------------------------------------------------

    print('Constructing the NN model...')

    padding = 'left'
    nb_filters = 64
    rnn_size = word_dims

    print('Building the NN computational graph...')

    words_input = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words')

    convs = []

    repr_size = 0


    if NET_ARCH == 'cnn':
        # сверточные слои
        for kernel_size in range(2, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            conv_layer = conv(words_input)
            conv_layer = GlobalMaxPooling1D()(conv_layer)
            convs.append(conv_layer)
            repr_size += nb_filters

    elif NET_ARCH == 'lstm+cnn':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        encoder_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                   input_shape=(max_inputseq_len, word_dims),
                                                   return_sequences=False))(words_input)

        convs.append(encoder_rnn)
        repr_size += rnn_size * 2

        # добавляем входы со сверточными слоями
        for kernel_size in range(2, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            conv_layer = conv(words_input)
            conv_layer = GlobalMaxPooling1D()(conv_layer)
            convs.append(conv_layer)
            repr_size += nb_filters

    elif NET_ARCH == 'cnn*lstm':
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

            conv_layer = conv(words_input)
            conv_layer = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')(
                conv_layer)
            conv_layer = lstm(conv_layer)
            convs.append(conv_layer)
            repr_size += rnn_size

    else:
        raise NotImplementedError('NET_ARCH={} is not implemented'.format(NET_ARCH))

    print('repr_size={}'.format(repr_size))
    encoder_size = repr_size
    encoder_merged = keras.layers.concatenate(inputs=convs)

    # финальный классификатор - грамматическое лицо фразы
    output_dims = 3
    encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_merged)
    encoder_final = Dense(rnn_size, activation='relu')(encoder_final)
    encoder_final = Dense(rnn_size//2, activation='relu')(encoder_final)
    encoder_final = Dense(output_dims, activation='softmax', name='output')(encoder_final)

    model = Model(inputs=words_input, outputs=encoder_final)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    keras.utils.plot_model(model,
                           to_file=os.path.join(tmp_folder, 'nn_person_classifier.{}.arch.png'.format(NET_ARCH)),
                           show_shapes=False,
                           show_layer_names=True)

    train_input, val_input, train_output, val_output = train_test_split(input_words,
                                                                        classif_ys,
                                                                        test_size=TEST_SHARE,
                                                                        random_state=SEED)

    nb_train_patterns = len(train_input)
    nb_valid_patterns = len(val_input)

    print('Train classifier using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_acc'

    model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
                                       verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=20, verbose=1, mode='auto')

    callbacks = [model_checkpoint, early_stopping]

    hist = model.fit_generator(generator=generate_rows_classifier(train_input, train_output, batch_size, 1),
                               steps_per_epoch=int(nb_train_patterns/batch_size),
                               epochs=200,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows_classifier(val_input, val_output, batch_size, 1),
                               validation_steps=int(nb_valid_patterns/batch_size)
                               )

if RUN_MODE=='query':

    with open( config_path, 'r') as f:
        model_config = json.load(f)

    max_inputseq_len = model_config['max_inputseq_len']
    word_dims = model_config['word_dims']

    print('0 - exit')
    print('1 - query person classifier')
    a1 = raw_input(':> ')
    if a1=='1':

        arch_filepath = os.path.join(tmp_folder, 'person_classifier.arch')
        weights_path = os.path.join(tmp_folder, 'person_classifier.weights')

        with open(arch_filepath, 'r') as f:
            model = model_from_json(f.read())

        model.load_weights(weights_path)

        while True:
            phrase = raw_input('phrase :> ').decode(sys.stdout.encoding).strip().lower()
            if len(phrase)==0:
                break

            X_batch = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)

            words = tokenizer.tokenize(phrase)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, X_batch, 0, word2vec )
            y = model.predict(X_batch)
            y = np.argmax( y[0] )
            y_str = {0:'1s', 1:'2s', 2:'3'}[y]
            print('person => {}'.format(y_str))


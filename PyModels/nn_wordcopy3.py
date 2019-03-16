# -*- coding: utf-8 -*-
"""
Нейросетевая модель для выбора цепочки слов, которые надо скопировать из предпосылки
для формирования ответа на вопрос.

Модель используется в проекте чат-бота https://github.com/Koziev/chatbot

Датасет должен быть предварительно сгенерирован скриптом prepare_qa_dataset.py

Кроме того, необходимо подготовить датасет с символьными встраиваниями слов, например
с помощью скрипта scripts/train_wordchar2vector.sh

Параметры обучения модели, в том числе детали архитектуры, задаются опциями
командной строки. Пример запуска обучения можно посмотреть в скрипте train_nn_wordcopy3.sh
"""

from __future__ import division
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
import argparse
import logging

import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.merge import concatenate, add, multiply
from keras.layers import Lambda
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json

from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers


#padding = 'left'
padding = 'right'
PAD_WORD = u''
CONFIG_FILENAME = 'nn_wordcopy3.config'


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова, чтобы длина стала равно n токенов"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def lpad_wordseq(words, n):
    """Слева добавляем пустые слова, чтобы длина стала равно n токенов"""
    return list(itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words))


def pad_wordseq(words, n):
    """Слева или справа добавляем пустые слова, чтобы длина стала равно n токенов"""
    if padding == 'right':
        return rpad_wordseq(words, n)
    else:
        return lpad_wordseq(words, n)


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def select_patterns(sequences, targets):
    sequences1 = []
    targets1 = []

    for seq, target in itertools.izip(sequences, targets):
        premise = seq[0]
        answer = target[0]

        all_words_mapped = True # все ли слова вопроса найдены в предпосылке?
        for word in answer:
            if word not in premise:
                all_words_mapped = False
                break

        if all_words_mapped:
            sequences1.append(seq)
            targets1.append(target)

    return sequences1, targets1


def count_words(words):
    return len(filter(lambda z: z != PAD_WORD, words))


def generate_rows_word_copy3(sequences0, targets0, batch_size, mode):
    '''Генератор данных в батчах для тренировки и валидации сетки'''
    assert(len(sequences0) == len(targets0))
    assert(0 < batch_size < 10000)
    assert(mode in[1, 2])

    batch_index = 0
    batch_count = 0

    n = len(sequences0)
    shuffled_indeces = list(np.random.permutation(range(n)))
    sequences = [sequences0[i] for i in shuffled_indeces]
    targets = [targets0[i] for i in shuffled_indeces]

    X1_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)

    y1_batch = np.zeros((batch_size, output_dims), dtype=np.bool)
    y2_batch = np.zeros((batch_size, output_dims), dtype=np.bool)

    while True:
        for irow, (seq, target) in enumerate(itertools.izip(sequences, targets)):
            vectorize_words(seq[0], X1_batch, batch_index, word2vec)
            vectorize_words(seq[1], X2_batch, batch_index, word2vec)

            premise = seq[0]
            answer = target[0]

            beg_found = False
            beg_pos = 0
            end_pos = 0
            for word in answer:
                if word in premise:
                    pos = premise.index(word)
                    if beg_found == False:
                        beg_found = True
                        beg_pos = pos
                    end_pos = max(end_pos, pos)
                else:
                    break

            assert(end_pos >= beg_pos)

            y1_batch[batch_index, beg_pos] = True
            y2_batch[batch_index, end_pos] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                x = {'input_words1': X1_batch, 'input_words2': X2_batch}
                if mode == 1:
                    yield (x, {'output_beg': y1_batch, 'output_end': y2_batch})
                else:
                    yield x

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                y1_batch.fill(0)
                y2_batch.fill(0)
                batch_index = 0

# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Neural model for word copy model for answer generation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture: lstm | lstm(cnn) | lstm+cnn | cnn')
parser.add_argument('--classifier', type=str, default='merge', help='final classifier architecture: merge | muladd')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/premise_question_answer.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.bin', help='path to word2vector model file')

args = parser.parse_args()
input_path = args.input
tmp_folder = args.tmp
run_mode = args.run_mode
batch_size = args.batch_size
net_arch = args.arch
classifier_arch = args.classifier
wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_wordcopy3.log'))

# В этих файлах будем сохранять натренированную сетку
arch_filepath = os.path.join(tmp_folder, 'nn_wordcopy3.arch')
weights_path = os.path.join(tmp_folder, 'nn_wordcopy3.weights')


if run_mode == 'train':
    logging.info('Start run_mode==train')

    max_inputseq_len = 0
    max_outputseq_len = 0  # максимальная длина ответа
    all_words = set()
    all_chars = set()

    logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    logging.info('wc2v_dims={0}'.format(wc2v_dims))

    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

    logging.info('samples.count={}'.format(df.shape[0]))

    tokenizer = Tokenizer()
    tokenizer.load()

    for i, record in df.iterrows():
        for phrase in [record['premise'], record['question']]:
            all_chars.update(phrase)
            words = tokenizer.tokenize(phrase)
            all_words.update(words)
            max_inputseq_len = max(max_inputseq_len, len(words))

        phrase = record['answer']
        all_chars.update(phrase)
        words = tokenizer.tokenize(phrase)
        all_words.update(words)
        max_outputseq_len = max(max_outputseq_len, len(words))

    for word in wc2v.vocab:
        all_words.add(word)
        all_chars.update(word)

    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_outputseq_len={}'.format(max_outputseq_len))

    word2id = dict(
        (c, i) for i, c in enumerate(itertools.chain([PAD_WORD], filter(lambda z: z != PAD_WORD, all_words))))

    nb_chars = len(all_chars)
    nb_words = len(all_words)
    logging.info('nb_chars={}'.format(nb_chars))
    logging.info('nb_words={}'.format(nb_words))

    # --------------------------------------------------------------------------

    logging.info(u'Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path,
                                                          binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    logging.info('w2v_dims={0}'.format(w2v_dims))

    word_dims = w2v_dims + wc2v_dims

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros(word_dims)
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    del w2v
    del wc2v
    gc.collect()

    logging.info('Constructing the NN model {} {}...'.format(net_arch, classifier_arch))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
        'engine': 'nn',
        'max_inputseq_len': max_inputseq_len,
        'max_outputseq_len': max_outputseq_len,
        'w2v_path': word2vector_path,
        'wordchar2vector_path': wordchar2vector_path,
        'PAD_WORD': PAD_WORD,
        'padding': padding,
        'model_folder': tmp_folder,
        'word_dims': word_dims
    }

    with open(os.path.join(tmp_folder, CONFIG_FILENAME), 'w') as f:
        json.dump(model_config, f)

    nb_filters = 128
    rnn_size = word_dims

    words_net1 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words1')
    words_net2 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words2')

    conv1 = []
    conv2 = []
    repr_size = 0

    if net_arch == 'lstm':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        repr_size = rnn_size  #*2
        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)

    if net_arch == 'lstm+cnn':
        # энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

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

            #dense2 = Dense(units=nb_filters)

            #pooler = GlobalMaxPooling1D()
            pooler = GlobalAveragePooling1D()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            #conv_layer1 = dense2(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            #conv_layer2 = dense2(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += nb_filters

    if net_arch == 'lstm(cnn)':
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

            #pooler = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')
            pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv_layer2 = lstm(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += rnn_size

    if net_arch == 'cnn':
        # простая сверточная архитектура.
        for kernel_size in range(1, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            # pooler = GlobalMaxPooling1D()
            pooler = GlobalAveragePooling1D()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += nb_filters

    # Тренируем модель, которая определяет позиции слов начала и конца цепочки.
    # Таким образом, у модели два независимых классификатора на выходе.
    output_dims = max_inputseq_len

    if classifier_arch == 'merge':
        encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2)))
        encoder_final = Dense(units=int(repr_size*2), activation='relu')(encoder_merged)

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
        #encoder1 = sent_repr_layer(encoder1)
        #encoder2 = sent_repr_layer(encoder2)
        sent2vec_dim = repr_size

        addition = add([encoder1, encoder2])
        minus_y1 = Lambda(lambda x: -x, output_shape=(sent2vec_dim,))(encoder1)
        mul = add([encoder2, minus_y1])
        mul = multiply([mul, mul])

        #words_final = keras.layers.concatenate(inputs=[encoder1, mul, addition, encoder2])
        encoder_final = keras.layers.concatenate(inputs=[mul, addition])

    else:
        logging.error('Unknown classifier arch: {}'.format(classifier_arch))

    decoder = Dense(rnn_size, activation='relu')(encoder_final)
    decoder = Dense(rnn_size//2, activation='relu')(decoder)
    decoder = Dense(rnn_size//3, activation='relu')(decoder)
    #decoder = Dense(rnn_size//4, activation='relu')(decoder)
    output_beg = Dense(output_dims, activation='softmax', name='output_beg')(decoder)

    decoder = Dense(rnn_size, activation='relu')(encoder_final)
    decoder = Dense(rnn_size//2, activation='relu')(decoder)
    decoder = Dense(rnn_size//3, activation='relu')(decoder)
    #decoder = Dense(rnn_size//4, activation='relu')(decoder)
    output_end = Dense(output_dims, activation='softmax', name='output_end')(decoder)

    model = Model(inputs=[words_net1, words_net2], outputs=[output_beg, output_end])
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.summary()

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    # -------------------------------------------------------------------------

    input_data = []
    output_data = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
        premise = row['premise']
        question = row['question']
        answer = row['answer']

        premise_words = pad_wordseq(tokenizer.tokenize(premise), max_inputseq_len)
        question_words = pad_wordseq(tokenizer.tokenize(question), max_inputseq_len)

        answer_words = tokenizer.tokenize(answer)
        input_data.append((premise_words, question_words, premise, question))
        output_data.append((answer_words, answer))

    SEED = 123456
    TEST_SHARE = 0.2
    train_input, val_input, train_output, val_output = train_test_split(input_data,
                                                                        output_data,
                                                                        test_size=TEST_SHARE,
                                                                        random_state=SEED)

    train_input1, train_output1 = select_patterns(train_input, train_output)
    val_input1, val_output1 = select_patterns(val_input, val_output)

    nb_train_patterns = len(train_input1)
    nb_valid_patterns = len(val_input1)

    logging.info('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_output_beg_acc'

    model_checkpoint = ModelCheckpoint(weights_path,
                                       monitor=monitor_metric,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=20, verbose=1, mode='auto')

    callbacks = [model_checkpoint, early_stopping]

    hist = model.fit_generator(generator=generate_rows_word_copy3(train_input1, train_output1, batch_size, 1),
                               steps_per_epoch=nb_train_patterns // batch_size,
                               epochs=200,
                               verbose=2,
                               callbacks=callbacks,
                               validation_data=generate_rows_word_copy3(val_input1, val_output1, batch_size, 1),
                               validation_steps=nb_valid_patterns // batch_size
                               )
    val_output_beg_acc = hist.history['val_output_beg_acc']
    val_output_end_acc = hist.history['val_output_end_acc']
    logging.info('max(val_output_beg_acc)={}'.format(max(val_output_beg_acc)))
    logging.info('max(val_output_end_acc)={}'.format(max(val_output_end_acc)))


if run_mode == 'query':
    # Ручная проверка работы натренированной модели.

    with open(os.path.join(tmp_folder, CONFIG_FILENAME), 'r') as f:
        model_config = json.load(f)

    max_inputseq_len = model_config['max_inputseq_len']
    max_outputseq_len = model_config['max_outputseq_len']
    word2vector_path = model_config['w2v_path']
    wordchar2vector_path = model_config['wordchar2vector_path']
    word_dims = model_config['word_dims']
    padding = model_config['padding']

    tokenizer = Tokenizer()

    print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    print('wc2v_dims={0}'.format(wc2v_dims))

    print('Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path,
                                                          binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))

    word_dims = w2v_dims + wc2v_dims

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros(word_dims)
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    del w2v
    del wc2v
    gc.collect()

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    X1_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)

    while True:
        print('\nEnter two phrases:')
        phrase1 = raw_input('premise :> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase1) == 0:
            break

        phrase2 = raw_input('question:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase2) == 0:
            break

        words1 = tokenizer.tokenize(phrase1)
        words2 = tokenizer.tokenize(phrase2)

        words1 = pad_wordseq(words1, max_inputseq_len)
        words2 = pad_wordseq(words2, max_inputseq_len)

        X1_probe.fill(0)
        X2_probe.fill(0)

        vectorize_words(words1, X1_probe, 0, word2vec)
        vectorize_words(words2, X2_probe, 0, word2vec)

        #for i1, word1 in enumerate(words1):
        #    if len(word1)>0:
        #        print(u'{} {} ==> {}'.format(i1, word1, X1_probe[0, i1, :]))

        y_probe = model.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})

        y1 = y_probe[0][0]
        y2 = y_probe[1][0]

        pos_beg = np.argmax(y1)
        pos_end = np.argmax(y2)

        print('pos_beg={} pos_end={}'.format(pos_beg, pos_end))

        answer_words = words1[pos_beg:pos_end+1]
        print(u'{}'.format(u' '.join(answer_words)))

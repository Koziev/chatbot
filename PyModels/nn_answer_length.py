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
import codecs
import tqdm
import argparse
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
import utils.console_helpers
import utils.logging_helpers



class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer


def count_words(words):
    """ Возвращает кол-во непустых строк в последовательности words """
    return len(filter(lambda z: z != PAD_WORD,words))


def pad_wordseq(words, n):
    """ Слева добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """ Справа добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain( words, itertools.repeat(PAD_WORD, n-len(words))))


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate( words ):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def generate_rows(nb_premises, samples, batch_size, mode):
    batch_index = 0
    batch_count = 0

    Xn_batch = []
    for _ in range(nb_premises+1):
        x = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_batch.append(x)

    inputs = {}
    for ipremise in range(nb_premises):
        inputs['premise{}'.format(ipremise)] = Xn_batch[ipremise]
    inputs['question'] = Xn_batch[nb_premises]

    y_batch = np.zeros((batch_size, output_dims), dtype=np.bool)

    weights = np.zeros((batch_size))
    weights.fill(1.0)

    while True:
        for irow, sample in enumerate(samples):
            for ipremise, premise in enumerate(sample.premises):
                words = tokenizer.tokenize(premise)
                words = pad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[ipremise], batch_index, word2vec)

            words = tokenizer.tokenize(sample.question)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[nb_premises], batch_index, word2vec)

            y_batch[batch_index, len(tokenizer.tokenize(sample.answer))-1] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield (inputs, {'output': y_batch}, weights)
                else:
                    yield inputs

                # очищаем матрицы порции для новой порции
                for x in Xn_batch:
                    x.fill(0)
                y_batch.fill(0)
                batch_index = 0



PAD_WORD = u''


parser = argparse.ArgumentParser(description='Neural model for answer length computation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture: lstm lstm(cnn)')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/pqa_all.dat', help='path to input dataset')
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
run_mode = args.run_mode


# В этих файлах будем сохранять натренированную сетку
arch_filepath = os.path.join(tmp_folder, 'nn_answer_length.arch')
weights_path = os.path.join(tmp_folder, 'nn_answer_length.weights')
config_path = os.path.join(tmp_folder, 'nn_answer_length.config')


if run_mode == 'train':

    max_inputseq_len = 0
    max_outputseq_len = 0 # максимальная длина ответа
    all_words = set()

    # --------------------------------------------------------------------------

    wordchar2vector_path = os.path.join(tmp_folder,'wordchar2vector.dat')
    print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    print('wc2v_dims={0}'.format(wc2v_dims))

    # --------------------------------------------------------------------------

    # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
    print(u'Loading samples from {}'.format(input_path))
    samples = []
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    tokenizer = Tokenizer()

    with codecs.open(input_path, 'r', 'utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if len(line) == 0:
                if len(lines) > 0:
                    premises = lines[:-2]
                    question = lines[-2]
                    answer = lines[-1]
                    sample = Sample(premises, question, answer)
                    samples.append(sample)

                    max_nb_premises = max(max_nb_premises, len(premises))

                    for phrase in lines:
                        words = tokenizer.tokenize(phrase)
                        all_words.update(words)
                        max_inputseq_len = max(max_inputseq_len, len(words))

                    lines = []

            else:
                lines.append(line)

    print('samples.count={}'.format(len(samples)))
    print('max_inputseq_len={}'.format(max_inputseq_len))
    print('max_nb_premises={}'.format(max_nb_premises))

    # Гистограмма длин ответов
    answerlen2count = Counter()
    answerlen2count.update(len(tokenizer.tokenize(sample.answer)) for sample in samples)

    for word in wc2v.vocab:
        all_words.add(word)

    print('max_inputseq_len={}'.format(max_inputseq_len))

    max_outputseq_len = max(len(tokenizer.tokenize(sample.answer)) for sample in samples)
    print('max_outputseq_len={}'.format(max_outputseq_len))

    word2id = dict((c, i) for (i, c) in enumerate(itertools.chain([PAD_WORD], filter(lambda z: z != PAD_WORD,all_words))))

    nb_words = len(all_words)
    print('nb_words={}'.format(nb_words))

    print('Count of answers by number of words:')
    for l, n in sorted(answerlen2count.iteritems(), key=lambda z: z[0]):
        print('number of words={:3} count={}'.format(l, n))

    # --------------------------------------------------------------------------

    print( 'Loading the w2v model {}'.format(word2vector_path) )
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
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
    # --------------------------------------------------------------------------------

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'engine': 'nn',
                    'max_inputseq_len': max_inputseq_len,
                    'max_outputseq_len': max_outputseq_len,
                    'max_nb_premises': max_nb_premises,
                    'word2vector_path': word2vector_path,
                    'wordchar2vector_path': wordchar2vector_path,
                    'PAD_WORD': PAD_WORD,
                    'model_folder': tmp_folder,
                    'word_dims': word_dims
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    # -------------------------------------------------------------------

    print('Constructing the NN model...')

    nb_filters = 128
    rnn_size = word_dims

    inputs = []
    input_question = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='question')
    inputs.append(input_question)

    for ipremise in range(max_nb_premises):
        input_premise = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='premise{}'.format(ipremise))
        inputs.append(input_premise)

    layers = []
    encoder_size = 0

    if net_arch == 'lstm':
        # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения. Этот слой общий для всех входных предложений.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        for input in inputs:
            encoder_rnn = shared_words_rnn(input)
            layers.append(encoder_rnn)
            encoder_size += rnn_size*2

    # --------------------------------------------------------------------------

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

            for input in inputs:
                conv_layer1 = conv(input)
                conv_layer1 = keras.layers.MaxPooling1D(pool_size=kernel_size,
                                                        strides=None,
                                                        padding='valid')(conv_layer1)
                conv_layer1 = lstm(conv_layer1)
                layers.append(conv_layer1)
                encoder_size += rnn_size

    encoder_merged = keras.layers.concatenate(inputs=list(layers))
    encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_merged)


    # финальный классификатор определяет длину ответа
    output_dims = max_outputseq_len
    decoder = Dense(rnn_size, activation='relu')(encoder_final)
    decoder = Dense(rnn_size//2, activation='relu')(decoder)
    decoder = Dense(output_dims, activation='softmax', name='output')(decoder)

    model = Model(inputs=inputs, outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    keras.utils.plot_model(model,
                           to_file=os.path.join(tmp_folder, 'nn_answer_length-{}.arch.png'.format(net_arch)),
                           show_shapes=False,
                           show_layer_names=True)


    SEED = 123456
    TEST_SHARE = 0.2
    train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)
    nb_train_patterns = len(train_samples)
    nb_valid_patterns = len(val_samples)

    print('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_acc'

    model_checkpoint = ModelCheckpoint(weights_path,
                                       monitor=monitor_metric,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.00001)

    callbacks = [reduce_lr, model_checkpoint, early_stopping]

    hist = model.fit_generator(generator=generate_rows(max_nb_premises, train_samples, batch_size, 1),
                               steps_per_epoch=nb_train_patterns//batch_size,
                               epochs=200,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(max_nb_premises, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns//batch_size
                               )

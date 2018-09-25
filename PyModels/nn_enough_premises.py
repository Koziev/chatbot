# -*- coding: utf-8 -*-
'''
Тренировка модели для определения достаточности набора предпосылок для ответа на вопрос.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
Используется нейросетка (Keras).
Датасет "pqa_all.dat" должен быть сгенерирован и находится в папке ../data (см. prepare_qa_dataset.py)
Также нужна модель встраивания слов (word2vector) и модель посимвольного встраивания (wordchar2vector).
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import argparse
import six
import codecs
from collections import Counter
import logging
import logging.handlers

import gensim
import keras.callbacks
import numpy as np
import tqdm
import sklearn.metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import initializers
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers import Flatten
from keras.layers.core import RepeatVector, Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import keras_contrib.optimizers.ftml

from utils.tokenizer import Tokenizer
from trainers.word_embeddings import WordEmbeddings

#from layers.word_match_layer import match


PAD_WORD = u''

padding = 'left'

# Кол-во ядер в сверточных слоях упаковщика предложений.
nb_filters = 128

initializer = 'random_normal'


class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer
        self.y = None


def lpad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def pad_wordseq(words, n):
    """Слева или справа добавляем пустые слова, чтобы длина стала равно n токенов"""
    if padding == 'right':
        return rpad_wordseq(words, n)
    else:
        return lpad_wordseq(words, n)


def vectorize_words(words, M, irow, word2vec):
    for iword, word in enumerate(words):
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

    y_batch = np.zeros((batch_size), dtype=np.bool)

    while True:
        for sample in samples:
            for ipremise, premise in enumerate(sample.premises):
                words = tokenizer.tokenize(premise)
                words = pad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[ipremise], batch_index, word2vec)

            words = tokenizer.tokenize(sample.question)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[nb_premises], batch_index, word2vec)

            y_batch[batch_index] = sample.y

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield (inputs, {'output': y_batch})
                else:
                    yield inputs

                # очищаем матрицы порции для новой порции
                for x in Xn_batch:
                    x.fill(0)
                y_batch.fill(0)
                batch_index = 0



parser = argparse.ArgumentParser(description='Neural model for premises completeness detection')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='encoder architecture: lstm | lstm(cnn)')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/pqa_all.dat', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.bin', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')
parser.add_argument('--w2v_folder', type=str, default='~/polygon/w2v')

args = parser.parse_args()

data_folder = args.data_dir
input_path = args.input
tmp_folder = args.tmp
w2v_folder = os.path.expanduser(args.w2v_folder)

wordchar2vector_path = args.wordchar2vector
word2vector_path = os.path.expanduser(args.word2vector)
batch_size = args.batch_size
net_arch = args.arch
run_mode = args.run_mode

# настраиваем логирование в файл
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
lf = logging.FileHandler(os.path.join(tmp_folder, 'nn_enough_premises.log'), mode='w')

lf.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
lf.setFormatter(formatter)
logging.getLogger('').addHandler(lf)

# В этих файлах будем сохранять натренированную сетку
config_path = os.path.join(tmp_folder, 'nn_enough_premises.config')
arch_filepath = os.path.join(tmp_folder, 'nn_enough_premises.arch')
weights_path = os.path.join(tmp_folder, 'nn_enough_premises.weights')

# -------------------------------------------------------------------

tokenizer = Tokenizer()

if run_mode == 'train':
    logging.info('Start run_mode==train')

    wordchar2vector_path = os.path.join(data_folder, 'wordchar2vector.dat')
    logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    logging.info('wc2v_dims={0}'.format(wc2v_dims))

    # --------------------------------------------------------------------------
    # Загружаем датасет
    max_inputseq_len = 0
    all_words = set([PAD_WORD])

    logging.info(u'Loading samples from {}'.format(input_path))
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

    if False:
        # В датасете очень много сэмплов с ответом "да".
        # Оставим столько да-сэмплов, сколько есть нет-сэмплов.
        nb_no = sum((sample.answer == u'нет') for sample in samples)
        samples_yes = filter(lambda sample: sample.answer == u'да', samples)
        samples_yes = np.random.permutation(list(samples_yes))[:nb_no]

        samples1 = filter(lambda sample: sample.answer != u'да', samples)
        samples1.extend(samples_yes)
        samples = samples1

    # Теперь генерируем из этих сэмплов данные
    samples0 = samples
    samples = []
    for sample in samples0:
        nb_premises = len(sample.premises)
        for n in range(nb_premises+1):
            # Создаем копию сэмпла с возможно усеченным списком предпосылок.
            sample2 = Sample(sample.premises[:n], sample.question, sample.answer)

            # целевое значение y имеет значение True, если список предпосылок достаточен,
            # и False, если нужно больше предпосылок.
            sample2.y = n == nb_premises
            samples.append(sample2)

    logging.info(u'samples.count={}'.format(len(samples)))

    for word in wc2v.vocab:
        all_words.add(word)

    logging.info('max_inputseq_len={}'.format(max_inputseq_len))

    nb_words = len(all_words)
    logging.info('nb_words={}'.format(nb_words))

    # --------------------------------------------------------------------------

    logging.info('Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
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

    logging.info('Building neural net net_arch={}'.format(net_arch))

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

    # декодер классифицирует паттерны на 2 класса
    output_dims = 1
    decoder = encoder_merged
    decoder = Dense(units=encoder_size//20, activation='relu')(decoder)
    decoder = Dense(units=encoder_size//30, activation='relu')(decoder)
    decoder = Dense(units=1, activation='sigmoid', name='output')(decoder)

    model = Model(inputs=inputs, outputs=decoder)

    opt = 'nadam'
    #opt = keras_contrib.optimizers.FTML()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    model.summary()

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    SEED = 123456
    TEST_SHARE = 0.2
    train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)

    nb_train_patterns = len(train_samples)
    nb_valid_patterns = len(val_samples)

    logging.info('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_acc'
    model_checkpoint = ModelCheckpoint(weights_path,
                                       monitor=monitor_metric,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric,
                                   patience=10,
                                   verbose=1,
                                   mode='auto')

    callbacks = [model_checkpoint, early_stopping]

    hist = model.fit_generator(generator=generate_rows(max_nb_premises, train_samples, batch_size, 1),
                               steps_per_epoch=nb_train_patterns//batch_size,
                               epochs=1000,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(max_nb_premises, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns//batch_size
                               )

    logging.info('Training is finished.')
    best_acc = max(hist.history['val_acc'])
    logging.info('Best validation accuracy={}'.format(best_acc))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'max_inputseq_len': max_inputseq_len,
                    'w2v_path': word2vector_path,
                    'wordchar2vector_path': wordchar2vector_path,
                    'PAD_WORD': PAD_WORD,
                    'padding': padding,
                    'model_folder': tmp_folder,
                    'max_nb_premises': max_nb_premises,
                    'word_dims': word_dims,
                    'arch_filepath': arch_filepath,
                    'weights_path': weights_path,
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    model.load_weights(weights_path)

    logging.info('Final validation for {} samples...'.format(len(samples)))
    # Выполним валидацию всех сэмплов, сохраним результаты в файл для
    # визуального анализа.
    nb_patterns = len(samples)
    for v in generate_rows(max_nb_premises, samples, nb_patterns, 1):
        x = v[0]
        y_true = v[1]['output']
        break

    y_pred = model.predict(x)
    y_pred = (y_pred >= 0.5).astype(np.int)
    f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred)
    logging.info('val f1={}'.format(f1))

    # Сохраним в текстовом файле для визуальной проверки результаты валидации по всем сэмплам
    for v in generate_rows(max_nb_premises, samples, len(samples), 1):
        x = v[0]
        break

    y_pred = model.predict(x)
    y_pred = (y_pred >= 0.5).astype(np.int)

    with codecs.open(os.path.join(tmp_folder, 'nn_enough_premises.validation.txt'), 'w', 'utf-8') as wrt:
        for isample, sample in enumerate(samples):
            if isample > 0:
                wrt.write('\n\n')

            for premise in sample.premises:
                wrt.write(u'P: {}\n'.format(premise))
            wrt.write(u'Q: {}\n'.format(sample.question))
            wrt.write(u'A: {}\n'.format(sample.answer))
            wrt.write(u'y_true={}\n'.format(sample.y))
            wrt.write(u'model: {}\n\n\n'.format(y_pred[isample]))

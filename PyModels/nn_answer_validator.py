# -*- coding: utf-8 -*-
'''
Тренировка модели для определения валидности ответа для указанного вопроса.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import gensim
import codecs
import keras.callbacks
import numpy as np
import tqdm
import argparse
import random

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
import sklearn.metrics

from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers



PAD_WORD = u''
padding = 'right'


class Sample:
    def __init__(self, question, answer, y):
        self.question = question
        self.answer = answer
        self.y = y



def lpad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def vectorize_words(words, M, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def get_pading_func(padding):
    return lpad_wordseq if padding == 'left' else rpad_wordseq


def generate_rows(samples, max_inputseq_len, word_dims, batch_size, mode):
    batch_index = 0
    batch_count = 0

    Xn_batch = []
    for _ in range(2):
        x = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_batch.append(x)

    inputs = {}
    inputs['question'] = Xn_batch[0]
    inputs['answer'] = Xn_batch[1]

    y_batch = np.zeros((batch_size), dtype=np.bool)

    pad_func = get_pading_func(padding)

    while True:
        for irow, sample in enumerate(samples):
            words = tokenizer.tokenize(sample.question)
            words = pad_func(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[0], batch_index, word2vec)

            words = tokenizer.tokenize(sample.answer)
            words = pad_func(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[1], batch_index, word2vec)

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




# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Neural model for answer validity estimation')
parser.add_argument('--run_mode', type=str, default='train', choices='train query'.split(), help='what to do')
parser.add_argument('--arch', type=str, default='lstm', choices='lstm lstm(cnn)'.split(), help='neural model architecture')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/pqa_all.dat', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')
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
config_path = os.path.join(tmp_folder,'nn_answer_validator.config')
arch_filepath = os.path.join(tmp_folder, 'nn_answer_validator.arch')
weights_path = os.path.join(tmp_folder, 'nn_answer_validator.weights')

if run_mode == 'train':
    max_inputseq_len = 0
    all_words = set()

    wordchar2vector_path = os.path.join(tmp_folder, 'wordchar2vector.dat')
    print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])

    # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ,
    # из них берем только пары вопрос-ответ
    print(u'Loading samples from {}'.format(input_path))
    samples = []
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    tokenizer = Tokenizer()

    all_questions = set()
    all_answers = set()
    known_qa = set()

    with codecs.open(input_path, 'r', 'utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if len(line) == 0:
                if len(lines) > 0:
                    question = lines[-2]
                    answer = lines[-1]
                    sample = Sample(question, answer, 1)
                    samples.append(sample)

                    all_questions.add(question)
                    all_answers.add(answer)
                    known_qa.add(question+'|'+answer)

                    for phrase in [question, answer]:
                        words = tokenizer.tokenize(phrase)
                        all_words.update(words)
                        max_inputseq_len = max(max_inputseq_len, len(words))

                    lines = []

            else:
                lines.append(line)

    all_answers = list(all_answers)
    all_questions = list(all_questions)

    # В датасете очень много сэмплов с ответом "да".
    # Оставим столько да-сэмплов, сколько есть нет-сэмплов.
    nb_no = sum((sample.answer == u'нет') for sample in samples)
    samples_yes = filter(lambda sample: sample.answer == u'да', samples)
    samples_yes = np.random.permutation(list(samples_yes))[:nb_no]

    samples1 = filter(lambda sample: sample.answer != u'да', samples)
    samples1.extend(samples_yes)
    samples = samples1

    print('samples.count={}'.format(len(samples)))
    print('max_inputseq_len={}'.format(max_inputseq_len))

    for word in wc2v.vocab:
        all_words.add(word)

    word2id = dict([(c, i) for i, c in enumerate(itertools.chain([PAD_WORD], filter(lambda z: z != PAD_WORD,all_words)))])

    nb_words = len(all_words)
    print('nb_words={}'.format(nb_words))

    # Добавляем негативные сэмплы
    nb_negatives = 1
    neg_samples = []
    for sample in tqdm.tqdm(samples, total=len(samples), desc='Generating negative samples'):
        random_answer = random.choice(all_answers)
        qa = sample.question + '|' + random_answer
        if qa not in known_qa:
            bad_sample = Sample(sample.question, random_answer, 0)
            neg_samples.append(bad_sample)

    samples.extend(neg_samples)

    print('Total number of samples={}'.format(len(samples)))

    # --------------------------------------------------------------------------

    print('Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])

    word_dims = w2v_dims+wc2v_dims

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros(word_dims)
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
                    'w2v_path': word2vector_path,
                    'wordchar2vector_path': wordchar2vector_path,
                    'PAD_WORD': PAD_WORD,
                    'padding': padding,
                    'model_folder': tmp_folder,
                    'word_dims': word_dims,
                    'arch_filepath': arch_filepath,
                    'weights_filepath': weights_path
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    # -------------------------------------------------------------------

    print('Constructing neural net: {}...'.format(net_arch))

    nb_filters = 128
    rnn_size = word_dims

    final_merge_size = 0

    inputs = []
    input_question = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='question')
    inputs.append(input_question)

    input_answer = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='answer')
    inputs.append(input_answer)

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
    decoder = Dense(units=encoder_size, activation='relu')(encoder_merged)
    #decoder = Dense(units=encoder_size//2, activation='relu')(decoder)
    decoder = Dense(units=encoder_size//4, activation='relu')(decoder)
    #decoder = BatchNormalization()(decoder)
    decoder = Dense(units=1, activation='sigmoid', name='output')(decoder)

    model = Model(inputs=inputs, outputs=decoder)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

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
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=20, verbose=1, mode='auto')
    callbacks = [model_checkpoint, early_stopping]

    hist = model.fit_generator(generator=generate_rows(train_samples, max_inputseq_len, word_dims, batch_size, 1),
                               steps_per_epoch=nb_train_patterns//batch_size,
                               epochs=100,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(val_samples, max_inputseq_len, word_dims, batch_size, 1),
                               validation_steps=nb_valid_patterns//batch_size)
    print('max val_acc={}'.format(max(hist.history['val_acc'])))

    # Загрузим лучшие веса и прогоним валидационные паттерны через модель,
    # чтобы получить f1 score.
    model.load_weights(weights_path)

    for v in generate_rows(val_samples, max_inputseq_len, word_dims, nb_valid_patterns, 1):
        x = v[0]
        y_val = v[1]['output']
        break

    y_pred = model.predict(x)
    y_pred = (y_pred >= 0.5).astype(np.int)
    f1 = sklearn.metrics.f1_score(y_true=y_val, y_pred=y_pred)
    print('val f1={}'.format(f1))

if run_mode == 'query':
    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_inputseq_len = model_config['max_inputseq_len']
        w2v_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        padding = model_config['padding']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])

    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])

    word2vec = dict()
    for word in wc2v.vocab:
        v = np.zeros( word_dims )
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    del w2v
    gc.collect()

    tokenizer = Tokenizer()

    Xn_probe = []
    for _ in range(2):
        x = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_probe.append(x)

    inputs = dict()
    inputs['question'] = Xn_probe[0]
    inputs['answer'] = Xn_probe[1]

    pad_func = get_pading_func(padding)

    while True:
        question = utils.console_helpers.input_kbd('question:> ').lower()
        if len(question) == 0:
            break

        answer = utils.console_helpers.input_kbd('answer:> ').lower()
        if len(answer) == 0:
            break

        for i in range(2):
            Xn_probe[i].fill(0)

        words = tokenizer.tokenize(question)
        words = pad_func(words, max_inputseq_len)
        vectorize_words(words, Xn_probe[0], 0, word2vec)

        words = tokenizer.tokenize(answer)
        words = pad_func(words, max_inputseq_len)
        vectorize_words(words, Xn_probe[1], 0, word2vec)

        y_probe = model.predict(x=inputs)

        print('y={}'.format(y_probe[0]))

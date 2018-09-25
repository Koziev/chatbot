# -*- coding: utf-8 -*-
'''
Нейросетевая модель для посимвольной генерации ответа на вопрос, заданный
к определенной фразе-предпосылке. Генерируется полный ответ сразу (в виде цепочки символов).

Для проекта чат-бота https://github.com/Koziev/chatbot

Используемые датасеты должны быть предварительно сгенерированы
скриптом scripts/prepare_datasets.sh
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import argparse
import sys
import six
import codecs
from collections import Counter
import logging
import logging.handlers

import gensim
import keras.callbacks
import numpy as np
import pandas as pd
import tqdm
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

from layers.word_match_layer import match


PAD_WORD = u''
PAD_CHAR = u'\r'

padding = 'left'

# Максимальная длина ответа в символах.
MAX_ANSWER_LEN = 10

# Кол-во ядер в сверточных слоях упаковщика предложений.
nb_filters = 128

USE_WORD_MATCHING = False

initializer = 'random_normal'


class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer


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


def rpad_charseq(s, n):
    return s+PAD_CHAR*max(0, n-len(s))


def decode_ystr(y, index2char):
    s = []
    for char_v in y:
        char_index = np.argmax(char_v)
        c = index2char[char_index]
        s.append(c)

    return u''.join(s).replace(PAD_CHAR, u' ').strip()


def vectorize_words(words, M, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def generate_rows(nb_premises, samples, batch_size, mode, char2id):
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

    y_batch = np.zeros((batch_size, max_outputseq_len, output_dims), dtype=np.bool)

    while True:
        for irow, sample in enumerate(samples):
            for ipremise, premise in enumerate(sample.premises):
                words = tokenizer.tokenize(premise)
                words = pad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[ipremise], batch_index, word2vec)

            words = tokenizer.tokenize(sample.question)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[nb_premises], batch_index, word2vec)

            answer = rpad_charseq(sample.answer, max_outputseq_len)
            for ichar, c in enumerate(answer):
                y_batch[batch_index, ichar, char2id[c]] = True

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



parser = argparse.ArgumentParser(description='Neural model for char-by-char generation of answer')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='encoder architecture: lstm | lstm(cnn)')
parser.add_argument('--generator', type=str, default='lstm(lstm)', help='decoder architecture: lstm | lstm(lstm) | lstm(lstm(lstm))')
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
generator_arch = args.generator
run_mode = args.run_mode

# настраиваем логирование в файл
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
lf = logging.FileHandler(os.path.join(tmp_folder, 'nn_chargenerator.log'), mode='w')

lf.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
lf.setFormatter(formatter)
logging.getLogger('').addHandler(lf)

# В этих файлах будем сохранять натренированную сетку
config_path = os.path.join(tmp_folder, 'nn_chargenerator.config')
arch_filepath = os.path.join(tmp_folder, 'nn_chargenerator.arch')
weights_path = os.path.join(tmp_folder, 'nn_chargenerator.weights')


# -------------------------------------------------------------------


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class VisualizeCallback(keras.callbacks.Callback):

    def __init__(self, max_nb_premises, val_samples, model, weights_path, char2id):
        self.epoch = 0
        self.max_nb_premises = max_nb_premises
        self.val_samples = val_samples
        self.model = model
        self.weights_path = weights_path
        self.char2id = char2id
        self.id2char = dict((i, c) for (c, i) in char2id.items())
        self.best_acc = 0
        self.stop_epoch = 0
        self.early_stopping = 20
        self.wait_epoch = 0
        self.val_acc_history = []  # для сохранения кривой обучения

    def decode_str(self, y):
        return decode_ystr(y, self.id2char)

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        nval = len(self.val_samples)

        # восстанавливаем сгенерированные символьные цепочки ответов, сравниваем с
        # требуемыми цепочками.
        nb_errors = 0

        # Счетчик напечатанных строк, сгенерированных моделью
        nb_shown = 0

        nb_steps = nval//batch_size

        print('')
        for step, batch in enumerate(generate_rows(self.max_nb_premises, self.val_samples, batch_size, 1, self.char2id)):
            if step == nb_steps:
                break

            y_batch = batch[1]['output']
            y_pred = model.predict_on_batch(batch[0])

            for iy in range(len(y_pred)):
                target_chars = self.decode_str(y_batch[iy])
                predicted_chars = self.decode_str(y_pred[iy])

                if predicted_chars != target_chars:
                    nb_errors += 1

                if nb_shown < 10:
                    print(
                        colors.ok + '☑ ' + colors.close if predicted_chars == target_chars else colors.fail + '☒ ' + colors.close,
                        end='')

                    print(u'true={}\t\tmodel={}'.format(target_chars, predicted_chars))
                    nb_shown += 1

        acc = (nval-nb_errors)/float(nval)
        self.val_acc_history.append(acc)
        if acc > self.best_acc:
            print(colors.ok+'New best instance accuracy={}\n'.format(acc)+colors.close)
            self.wait_epoch = 0
            self.model.save_weights(self.weights_path)
            self.best_acc = acc
        else:
            self.wait_epoch += 1
            print('\nInstance accuracy={} did not improve ({} epochs since last improvement)\n'.format(acc, self.wait_epoch))
            if self.wait_epoch >= self.early_stopping:
                print('Training stopped after {} epochs without impromevent'.format(self.wait_epoch))
                print('Best instance accuracy={}'.format(self.best_acc))
                self.model.stop_training = True
                self.stop_epoch = self.epoch

    def save_learning_curve(self, path):
        with open(path, 'w') as wrt:
            wrt.write('epoch\tacc\n')
            for i, acc in enumerate(self.val_acc_history):
                wrt.write('{}\t{}\n'.format(i+1, acc))

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
    # Загружаем датасет, анализируем использование символов и слов.
    max_inputseq_len = 0
    max_outputseq_len = 0  # максимальная длина ответа
    all_words = set([PAD_WORD])
    all_chars = set([PAD_CHAR])

    logging.info(u'Loading samples from {}'.format(input_path))
    samples = []
    nb_yes = 0 # кол-во ответов "да"
    nb_no = 0 # кол-во ответов "нет"
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
                    if len(answer) <= MAX_ANSWER_LEN:
                        sample = Sample(premises, question, answer)
                        samples.append(sample)

                        max_nb_premises = max(max_nb_premises, len(premises))
                        max_outputseq_len = max(max_outputseq_len, len(answer))

                        for phrase in lines:
                            all_chars.update(phrase)
                            words = tokenizer.tokenize(phrase)
                            all_words.update(words)
                            max_inputseq_len = max(max_inputseq_len, len(words))

                    lines = []

            else:
                lines.append(line)

    # В датасете очень много сэмплов с ответом "да".
    # Оставим столько да-сэмплов, сколько есть нет-сэмплов.
    nb_no = sum((sample.answer == u'нет') for sample in samples)
    samples_yes = filter(lambda sample: sample.answer == u'да', samples)
    samples_yes = np.random.permutation(list(samples_yes))[:nb_no]

    samples1 = filter(lambda sample: sample.answer != u'да', samples)
    samples1.extend(samples_yes)
    samples = samples1

    for word in wc2v.vocab:
        all_words.add(word)

    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_outputseq_len={}'.format(max_outputseq_len))

    char2id = dict(
        (c, i) for i, c in enumerate(itertools.chain([PAD_CHAR], filter(lambda z: z != PAD_CHAR, all_chars))))

    nb_chars = len(all_chars)
    nb_words = len(all_words)
    logging.info('nb_chars={}'.format(nb_chars))
    logging.info('nb_words={}'.format(nb_words))

    # --------------------------------------------------------------------------

    logging.info(u'Loading the w2v model {}'.format(word2vector_path))
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
    # -------------------------------------------------------------------

    rnn_size = word_dims

    inputs = []
    input_question = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='question')
    inputs.append(input_question)

    for ipremise in range(max_nb_premises):
        input_premise = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='premise{}'.format(ipremise))
        inputs.append(input_premise)

    layers = []
    encoder_size = 0

    logging.info('Building neural net net_arch={}'.format(net_arch))

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

    # декодер генерирует цепочку символов ответа
    output_dims = nb_chars
    decoder = encoder_merged
#    decoder = Dense(units=encoder_size, activation='relu')(decoder)
#    decoder = Dense(units=encoder_size, activation='relu')(decoder)
#    decoder = Dense(units=encoder_size, activation='relu')(decoder)
    decoder = RepeatVector(max_outputseq_len)(decoder)

    # Стек из нескольких рекуррентных слоев. Предполагается, что первый
    # слой формирует грубую структуру ответа, а второй слой уточняет
    # ее до точной цепочки символов и т.д., а последний слой формирует
    # цепочку символов
    if generator_arch == 'lstm(lstm(lstm))':
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
    elif generator_arch == 'lstm(lstm)':
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
    else:
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)

    decoder = TimeDistributed(Dense(nb_chars, activation='softmax', kernel_initializer=initializer), name='output')(decoder)

    model = Model(inputs=inputs, outputs=decoder)

    #opt = 'nadam'
    #opt = 'rmsprop'
    opt = 'nadam'
    #opt = keras_contrib.optimizers.FTML()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    SEED = 123456
    TEST_SHARE = 0.2
    train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)

    nb_train_patterns = len(train_samples)
    nb_valid_patterns = len(val_samples)

    logging.info('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    #monitor_metric = 'val_loss'
    #model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric, verbose=1, save_best_only=True, mode='auto')
    #early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    viz = VisualizeCallback(max_nb_premises, val_samples, model, weights_path, char2id)

    # callbacks = [model_checkpoint, early_stopping, viz]
    callbacks = [viz]

    hist = model.fit_generator(generator=generate_rows(max_nb_premises, train_samples, batch_size, 1, char2id),
                               steps_per_epoch=nb_train_patterns//batch_size,
                               epochs=1000,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(max_nb_premises, val_samples, batch_size, 1, char2id),
                               validation_steps=nb_valid_patterns//batch_size
                               )

    logging.info('Training is finished.')
    logging.info('Best instance accuracy={}'.format(viz.best_acc))

    viz.save_learning_curve(os.path.join(tmp_folder, 'nn_chargenerator.learning_curve.tsv'))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'max_inputseq_len': max_inputseq_len,
                    'max_outputseq_len': max_outputseq_len,
                    'w2v_path': word2vector_path,
                    'wordchar2vector_path': wordchar2vector_path,
                    'PAD_WORD': PAD_WORD,
                    'PAD_CHAR': ord(PAD_CHAR),
                    'model_folder': tmp_folder,
                    'max_nb_premises': max_nb_premises,
                    'word_dims': word_dims,
                    'arch_filepath': arch_filepath,
                    'weights_path': weights_path,
                    'char2id': char2id
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    model.load_weights(weights_path)

    if False:
        # Сохраним кодирующую часть как отдельную сетку.
        encoder_model = Model(inputs=[words_net1, words_net2], outputs=encoder_final)
        encoder_model.compile(loss='categorical_crossentropy', optimizer=opt)

        arch_filepath2 = os.path.join(tmp_folder, 'qa_chargenerator_model.sent2vector.arch')
        weights_path2 = os.path.join(tmp_folder, 'qa_chargenerator_model.sent2vector.weights')

        with open(arch_filepath2, 'w') as f:
            f.write(encoder_model.to_json())
        encoder_model.save_weights(weights_path2)

    nval = len(val_samples)
    logging.info(u'Финальная валидация модели на {} сэмплах'.format(nval))
    id2char = dict((i, c) for c, i in char2id.items())

    # Накопим кол-во ошибок и сэмплов для ответов разной длины.
    answerlen2samples = Counter()
    answerlen2errors = Counter()

    with codecs.open(os.path.join(tmp_folder, 'nn_chargenerator_model.validation.txt'), 'w', 'utf-8') as wrt:
        nb_steps = nval // batch_size
        isample = 0
        for step, batch in enumerate(generate_rows(max_nb_premises, val_samples, batch_size, 1, char2id)):
            if step == nb_steps:
                break

            y_batch = batch[1]['output']
            y_pred = model.predict_on_batch(batch[0])

            for iy in range(len(y_pred)):
                target_chars = decode_ystr(y_batch[iy], id2char)
                predicted_chars = decode_ystr(y_pred[iy], id2char)

                answer_len = len(target_chars)
                answerlen2samples[answer_len] += 1

                if predicted_chars != target_chars:
                    for ipremise, premise in enumerate(val_samples[isample].premises):
                        wrt.write(u'Premise[{}]:  {}\n'.format(ipremise, premise))
                    wrt.write(u'Question:     {}\n'.format(val_samples[isample].question))
                    wrt.write(u'True answer:  {}\n'.format(target_chars))
                    wrt.write(u'Model answer: {}\n'.format(predicted_chars))
                    wrt.write('\n\n')
                    answerlen2errors[answer_len] += 1

                isample += 1

    # Accuracy for answers with respect to their lengths:
    with open(os.path.join(tmp_folder, 'nn_chargenerator_model.accuracy.csv'), 'w') as wrt:
        wrt.write('answer_len\tnb_samples\taccuracy\n')
        for answer_len in sorted(answerlen2samples.keys()):
            support = answerlen2samples[answer_len]
            nb_err = answerlen2errors[answer_len]
            acc = 1.0 - float(nb_err)/float(support)
            wrt.write(u'{}\t{}\t{}\n'.format(answer_len, support, acc))


if run_mode == 'query':
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    max_inputseq_len = cfg['max_inputseq_len']
    max_outputseq_len = cfg['max_outputseq_len']
    max_nb_premises = cfg['max_nb_premises']
    w2v_path = cfg['w2v_path']
    wordchar2vector_path = cfg['wordchar2vector_path']
    word_dims = cfg['word_dims']
    arch_filepath = cfg['arch_filepath']
    weights_path = cfg['weights_path']
    char2id = cfg['char2id']

    w2v_path = os.path.join(w2v_folder, os.path.basename(w2v_path))

    index2char = dict((i, c) for (c, i) in six.iteritems(char2id))

    print('Restoring model architecture from {}'.format(arch_filepath))
    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    print('Loading model weights from {}'.format(weights_path))
    model.load_weights(weights_path)

    print('Loading word embeddings')
    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, w2v_path)
    word_dims = embeddings.vector_size

    Xn_probe = []
    for _ in range(max_nb_premises+1):
        x = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_probe.append(x)

    inputs = dict()
    for ipremise in range(max_nb_premises):
        inputs['premise{}'.format(ipremise)] = Xn_probe[ipremise]
    inputs['question'] = Xn_probe[max_nb_premises]

    while True:
        print('\nEnter 0 to {} premises and one question:'.format(max_nb_premises))
        premises = []
        question = None
        for ipremise in range(max_nb_premises):
            premise = raw_input('premise #{} :> '.format(ipremise)).decode(sys.stdout.encoding).strip().lower()
            if len(premise) == 0:
                break
            if premise[-1] == u'?':
                question = premise
                break

            premises.append(premise)

        if question is None:
            question = raw_input('question:> ').decode(sys.stdout.encoding).strip().lower()
        if len(question) == 0:
            break

        for i in range(max_nb_premises+1):
            Xn_probe[i].fill(0)

        for ipremise, premise in enumerate(premises):
            words = tokenizer.tokenize(premise)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_probe[ipremise], 0, embeddings)

        words = tokenizer.tokenize(question)
        words = pad_wordseq(words, max_inputseq_len)
        vectorize_words(words, Xn_probe[max_nb_premises], 0, embeddings)

        y = model.predict(x=inputs)

        answer = decode_ystr(y[0], index2char)

        print(u'{}'.format(answer))

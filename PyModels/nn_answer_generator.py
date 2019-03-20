# -*- coding: utf-8 -*-
'''
Тренировка модели, которая посимвольно в режиме teacher forcing учится генерировать
ответ для заданной предпосылки и вопроса.

В качестве классификационного движка для выбора символов используется нейросетка

За один запуск модели выбирается один новый символ, который добавляется к ранее сгенерированной
цепочке символов ответа (см. функцию generate_answer). Генерация через повторные запуски продолжается
до появления специального маркера конца цепочки END_CHAR.

В роли символов могут выступать также более крупные фрагменты слова - слоги или n-граммы из SentencePiece
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import itertools
import json
import os
import sys
import argparse
import codecs
import gzip
from collections import Counter
import six
import numpy as np
import pandas as pd
import tqdm

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import gensim
import keras
from keras.layers import Lambda
from keras.layers.merge import add, multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers import Flatten
from keras.layers.core import RepeatVector, Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.models import model_from_json
import sentencepiece as spm

import rusyllab

from utils.tokenizer import Tokenizer
import utils.console_helpers
from trainers.word_embeddings import WordEmbeddings


# -------------------------------------------------------------------

input_path = '../data/premise_question_answer.csv'
tmp_folder = '../tmp'
data_folder = '../data'
w2v_path = os.path.expanduser('~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin')
#w2v_path = os.path.expanduser('~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=100.bin')

# -------------------------------------------------------------------


answer_representation = 'chars'
#answer_representation = 'syllables'
#answer_representation = 'sentencepiece'

BATCH_SIZE = 300

# Некоторые паттерны содержат очень длинные ответы, длиннее 100 символов, исключим их.
MAX_ANSWEWR_LEN = 20

# Сэмплы с числом слов более заданного в предпосылке или вопросе исключим
MAX_PREMISE_LEN = 10

# Кол-во ядер в сверточных слоях упаковщика предложений.
nb_filters = 64

NET_ARCH = 'lstm'
#NET_ARCH = 'lstm(cnn)'
#NET_ARCH = 'cnn'


BEG_WORD = u'\b'
END_WORD = u'\n'

BEG_CHAR = u'\b'
END_CHAR = u'\n'

PAD_WORD = u''
PAD_CHAR = u'\r'


spm_encoder = spm.SentencePieceProcessor()
spm_encoder.Load('../tmp/spm_answers_model.model')


# -------------------------------------------------------------------


def answer2pieces(answer_str, max_answer_len):
    if answer_representation == 'chars':
        # вариант для разбивки на символы
        return rpad_chars(BEG_CHAR + answer_str + END_CHAR, max_answer_len)
    elif answer_representation == 'syllables':
        # вариант для разбивки на слоги
        seq = [BEG_CHAR] + rusyllab.split_words(answer_str.split()) + [END_CHAR]
        l = len(seq)
        if l < max_answer_len:
            seq = seq + list(itertools.repeat(PAD_CHAR, (max_answer_len - l)))
        return seq
    elif answer_representation == 'sentencepiece':
        seq = [BEG_CHAR] + spm_encoder.EncodeAsPieces(answer_str) + [END_CHAR]
        l = len(seq)
        if l < max_answer_len:
            seq = seq + list(itertools.repeat(PAD_CHAR, (max_answer_len - l)))
        return seq
    else:
        raise NotImplementedError()


def words2str(words):
    """
    Цепочку слов соединяем в строку, добавляя перед цепочкой и после нее
    пробел и специальные символы начала и конца.
    :param words:
    :return:
    """
    return BEG_WORD + u' ' + u' '.join(words) + u' ' + END_WORD


def undress(s):
    return s.replace(BEG_CHAR, u' ').replace(END_CHAR, u' ').strip()


def encode_char(c):
    if c == BEG_CHAR:
        return u'\\b'
    elif c == END_CHAR:
        return u'\\n'
    elif c == PAD_CHAR:
        return u'\\r'
    else:
        return c


def lpad_words(words, n):
    l = len(words)
    if l >= n:
        return words
    else:
        return list(itertools.chain(itertools.repeat(PAD_WORD, n-l), words))


def lpad_chars(chars, n):
    l = len(chars)
    if l >= n:
        return chars
    else:
        return list(itertools.chain(itertools.repeat(PAD_CHAR, n-l), chars))


def rpad_chars(chars, n):
    l = len(chars)
    if l >= n:
        return chars
    else:
        return list(itertools.chain(chars, itertools.repeat(PAD_CHAR, n-l)))


class Word2Lemmas(object):
    def __init__(self):
        pass

    def load(self, path):
        print('Loading lexicon from {}'.format(path))
        self.lemmas = dict()
        self.forms = dict()
        with gzip.open(path, 'r') as rdr:
            for line in rdr:
                tx = line.strip().decode('utf8').split('\t')
                if len(tx) == 2:
                    form = tx[0]
                    lemma = tx[1]

                    if form not in self.forms:
                        self.forms[form] = [lemma]
                    else:
                        self.forms[form].append(lemma)

                    if lemma not in self.lemmas:
                        self.lemmas[lemma] = {form}
                    else:
                        self.lemmas[lemma].add(form)
        print('Lexicon loaded: {} lemmas, {} wordforms'.format(len(self.lemmas), len(self.forms)))

    def get_forms(self, word):
        if word in self.forms:
            #result = set()
            #for lemma in self.forms[word]:
            #    result.update(self.lemmas[lemma])
            #return result
            return set(itertools.chain(*(self.lemmas[lemma] for lemma in self.forms[word])))
        else:
            return [word]



def generate_samples(premises, questions, answers, max_answer_len):
    inputs = []
    targets = []

    for premise, question, answer0 in itertools.izip(premises, questions, answers):
        answer = answer2pieces(answer0, max_answer_len)
        for answer_len in range(1, len(answer)):
            previous_chars = answer[:answer_len]
            target_char = answer[answer_len]
            inputs.append((premise, question, previous_chars))
            #if len(previous_chars) == 23:
            #    print('DEBUG@214')
            targets.append(target_char)

    return inputs, targets


def vectorize_words(words, M, irow, word2vec, word_dim):
    for iword, word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :word_dim] = word2vec[word]
            M[irow, iword, word_dim] = 1.0  # отметка "слово есть"
        else:
            M[irow, iword, word_dim] = 0.0  # "заполнитель"


def generate_rows(sequences, targets, batch_size, mode, max_inputseq_len, max_prevchars, char2id, word2vec, word_dim):
    batch_index = 0
    batch_count = 0
    nb_chars = len(char2id)

    X1_batch = np.zeros((batch_size, max_inputseq_len, word_dim+1), dtype=np.float32)  # слова предпосылки
    X2_batch = np.zeros((batch_size, max_inputseq_len, word_dim+1), dtype=np.float32)  # слова вопроса
    #X3_batch = np.zeros((batch_size, max_prevchars*nb_chars), dtype=np.float32)  # последние сгенерированные символы ответа
    X3_batch = np.zeros((batch_size, max_prevchars, nb_chars), dtype=np.float32)  # последние сгенерированные символы ответа

    y_batch = np.zeros((batch_size, nb_chars), dtype=np.bool)  # новый символ ответа

    while True:
        for irow, (seq, target_char) in enumerate(itertools.izip(sequences, targets)):
            vectorize_words(lpad_words(seq[0], max_inputseq_len), X1_batch, batch_index, word2vec, word_dim)
            vectorize_words(lpad_words(seq[1], max_inputseq_len), X2_batch, batch_index, word2vec, word_dim)

            #for ichar, c in enumerate(rpad_chars(seq[2][::-1], max_prevchars)):
            #    X3_batch[batch_index, ichar*nb_chars + char2index[c]] = 1.0
            for ichar, c in enumerate(rpad_chars(seq[2][::-1], max_prevchars)):
                X3_batch[batch_index, ichar, char2index[c]] = 1.0

            y_batch[batch_index, char2id[target_char]] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                x = {'input_premise': X1_batch,
                     'input_question': X2_batch,
                     'input_prevchars': X3_batch,
                     }

                if mode == 1:
                    yield (x, {'output': y_batch})
                else:
                    yield x

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                X3_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Answer text generator')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')

args = parser.parse_args()

run_mode = args.run_mode

tokenizer = Tokenizer()
tokenizer.load()

config_path = os.path.join(tmp_folder,'nn_answer_generator.config')

# В этих файлах будем сохранять натренированную сетку
arch_filepath = os.path.join(tmp_folder, 'nn_answer_generator.arch')
weights_path = os.path.join(tmp_folder, 'nn_answer_generator.weights')

if run_mode == 'train':
    # Загружаем и готовим датасет
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    print('samples.count={}'.format(df.shape[0]))

    all_chars = set()
    max_phrase_len = 0

    premises = []
    questions = []
    answers = []
    all_chars = set([PAD_CHAR])
    max_answer_len = 0

    for i, record in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Loading samples'):
        premise = record['premise']
        question = record['question']
        answer = record['answer'].lower()

        if answer not in [u'да'] and len(answer) <= MAX_ANSWEWR_LEN:
            premise_words = tuple(tokenizer.tokenize(premise))
            question_words = tuple(tokenizer.tokenize(question))

            if len(premise_words) <= MAX_PREMISE_LEN and len(question_words) <= MAX_PREMISE_LEN:
                answer_words = tokenizer.tokenize(answer)
                answer_str = u' '.join(answer_words)
                premises.append(premise_words)
                questions.append(question_words)
                answers.append(answer_str)

                pieces = answer2pieces(answer_str, 0)
                max_answer_len = max(max_answer_len, len(pieces))
                all_chars.update(pieces)

    nb_chars = len(all_chars)
    char2index = {PAD_CHAR: 0}
    for c in all_chars:
        if c != PAD_CHAR:
            char2index[c] = len(char2index)

    index2char = dict((i, c) for (c, i) in char2index.items())
    print('nb_chars={}'.format(nb_chars))
    print('max_answer_len={}'.format(max_answer_len))

    # Максимальная длина входных последовательностей.
    # Для предпосылки и вопроса это число слов.
    max_phrase_len = max(itertools.chain(map(len, premises), map(len, questions)))
    max_prevchars = max_answer_len
    print('max_phrase_len={}'.format(max_phrase_len))

    SEED = 123456
    TEST_SHARE = 0.2
    premises_train, premises_test,\
    questions_train, questions_test,\
    answers_train, answers_test = train_test_split(premises, questions, answers,
                                                   test_size=TEST_SHARE,
                                                   random_state=SEED)

    print('Generating training samples...')
    train_inputs, train_targets = generate_samples(premises_train, questions_train, answers_train, max_answer_len)
    nb_train = len(train_inputs)
    print('nb_train={}'.format(nb_train))

    print('Generating test samples...')
    test_inputs, test_targets = generate_samples(premises_test, questions_test, answers_test, max_answer_len)
    nb_test = len(test_inputs)
    print('nb_test={}'.format(nb_test))

    wc2v_path = os.path.join(data_folder, 'wordchar2vector.dat')
    word2vec = WordEmbeddings.load_word_vectors(wc2v_path, w2v_path)
    word_dims = word2vec.vector_size
    print('word_dims={0}'.format(word_dims))

    model_config = {
                    'engine': 'nn',
                    'max_inputseq_len': max_phrase_len,
                    'max_outseq_len': max_answer_len,
                    'w2v_path': w2v_path,
                    'wordchar2vector_path': wc2v_path,
                    'PAD_WORD': PAD_WORD,
                    'model_folder': tmp_folder,
                    'word_dims': word_dims,
                    'char2index': char2index,
                    'arch_filepath': arch_filepath,
                    'weights_filepath': weights_path
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=4)

    # Создаем нейросетку.
    rnn_size = word_dims*2

    final_merge_size = 0  # вычисляемый параметр сетки - размер вектора на выходе энкодера

    print('Building the NN computational graph {}'.format(NET_ARCH))
    input_premise = Input(shape=(max_phrase_len, word_dims+1,), dtype='float32', name='input_premise')
    input_question = Input(shape=(max_phrase_len, word_dims+1,), dtype='float32', name='input_question')
    #input_prevchars = Input(shape=(max_prevchars*nb_chars,), dtype='float32', name='input_prevchars')
    input_prevchars = Input(shape=(max_prevchars, nb_chars,), dtype='float32', name='input_prevchars')

    # порядковый номер выбираемого символа
    #input_charcount = Input(shape=(max_prevchars,), dtype='float32', name='input_charcount')

    merging_layers = []
    encoder_size = 0

    if NET_ARCH == 'cnn':
        for kernel_size in range(1, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            # pooler = keras.layers.GlobalAveragePooling1D()
            pooler = keras.layers.GlobalMaxPooling1D()

            conv_layer1 = conv(input_premise)
            conv_layer1 = pooler(conv_layer1)
            merging_layers.append(conv_layer1)
            encoder_size += nb_filters

            conv_layer2 = conv(input_question)
            conv_layer2 = pooler(conv_layer2)
            merging_layers.append(conv_layer2)
            encoder_size += nb_filters

    elif NET_ARCH == 'lstm(cnn)':
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
            pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')

            conv_layer1 = conv(input_premise)
            conv_layer1 = pooler(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            merging_layers.append(conv_layer1)
            encoder_size += rnn_size

            conv_layer2 = conv(input_question)
            conv_layer2 = pooler(conv_layer2)
            conv_layer2 = lstm(conv_layer2)
            merging_layers.append(conv_layer2)
            encoder_size += rnn_size

    elif NET_ARCH == 'lstm':
        lstm1 = Bidirectional(recurrent.LSTM(rnn_size, return_sequences=False))
        conv_layer1 = lstm1(input_premise)
        merging_layers.append(conv_layer1)
        encoder_size += rnn_size*2

        lstm2 = Bidirectional(recurrent.LSTM(rnn_size, return_sequences=False))
        conv_layer2 = lstm2(input_question)
        merging_layers.append(conv_layer2)
        merging_layers.append(conv_layer2)
        encoder_size += rnn_size*2

    # Отдельно работаем с цепочкой ранее выбранных символов
    #prev_chars = Dense(units=100, activation='relu')(input_prevchars)
    #encoder_size += 100

    prev_chars = recurrent.LSTM(100, return_sequences=False)(input_prevchars)
    merging_layers.append(prev_chars)
    encoder_size += 100


    # Все входные потоки объединяем
    encoder_merged = keras.layers.concatenate(inputs=merging_layers)
    encoder_final = encoder_merged
    encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_final)
    #encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_final)
    #encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_final)

    # декодер выбирает один следующий символ
    output_dims = nb_chars
    decoder = Dense(units=output_dims, activation='softmax', name='output')(encoder_final)

    model = Model(inputs=[input_premise, input_question, input_prevchars], outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])

    model.summary()

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    batch_size = BATCH_SIZE

    if True:
        print('Start training using {} patterns for training, {} for validation...'.format(nb_train, nb_test))

        monitor_metric = 'val_acc'
        model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric, verbose=1, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

        callbacks = [model_checkpoint, early_stopping]

        hist = model.fit_generator(generator=generate_rows(train_inputs, train_targets, batch_size, 1, max_phrase_len, max_prevchars, char2index, word2vec, word_dims),
                                   steps_per_epoch=nb_train // batch_size,
                                   epochs=1000,  #1000,
                                   verbose=1,
                                   callbacks=callbacks,
                                   validation_data=generate_rows(test_inputs, test_targets, batch_size, 1, max_phrase_len, max_prevchars, char2index, word2vec, word_dims),
                                   validation_steps=nb_test // batch_size
                                   )

        print('Training is finished.')

    # Сделаем финальную валидацию.
    # Для удобства сравнения реультатов разных моделей сегментации строк будем
    # оценивать точность per instance. Для этого каждый исходный сэмпл обрабатываем отдельно.
    nb_instance_errors = 0
    nb_instances = 0
    print('Final validation using {} samples'.format(nb_test))
    model.load_weights(weights_path)

    with codecs.open(os.path.join(tmp_folder, 'nn_answer_generator.validation.txt'), 'w', 'utf-8') as wrt:
        for premise, question, answer in tqdm.tqdm(itertools.izip(premises_test, questions_test, answers_test),
                                                   total=len(premises_test),
                                                   desc='Evaluation'):
            test_inputs, test_targets = generate_samples([premise], [question], [answer], max_answer_len)

            wrt.write(50*'-'+'\n')
            wrt.write(u'premise      ={}\n'.format(u' '.join(premise)))
            wrt.write(u'question     ={}\n'.format(u' '.join(question)))
            wrt.write(u'model answer ={}\n'.format(answer))

            sample_has_error = False
            predicted_answer = u''
            for batch in generate_rows(test_inputs, test_targets, len(test_targets), 1, max_phrase_len, max_prevchars, char2index, word2vec, word_dims):
                y_batch = batch[1]['output']
                y_pred = model.predict_on_batch(batch[0])

                y_batch = np.argmax(y_batch, axis=-1)
                y_pred = np.argmax(y_pred, axis=-1)

                for i, (yi_batch, yi_pred) in enumerate(itertools.izip(y_batch, y_pred)):
                    if yi_batch != yi_pred:
                        sample_has_error = True

                    #wrt.write(u'Prev chars: {}\n'.format(test_inputs[i][2]))
                    #wrt.write(u'True next char:  {}\n'.format(encode_char(index2char[yi_batch])))
                    #wrt.write(u'Model next char: {}\n'.format(encode_char(index2char[yi_pred])))
                    predicted_answer += index2char[yi_pred]

                break

            if predicted_answer[-1] == END_CHAR:
                predicted_answer = predicted_answer[:-1]

            predicted_answer = undress(predicted_answer)

            wrt.write(u'pred. answer ={}'.format(predicted_answer))
            nb_instances += 1
            if sample_has_error:
                nb_instance_errors += 1
                wrt.write(u'  <-- ERROR!\n')
            else:
                wrt.write(u'\n')

    acc = 1.0 - float(nb_instance_errors) / nb_instances
    print('Accuracy per instance={}'.format(acc))


if run_mode == 'query':
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_inputseq_len = model_config['max_inputseq_len']
        max_outseq_len = model_config['max_outseq_len']
        w2v_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        #padding = model_config['padding']
        char2index = model_config['char2index']
        #max_nb_inputs = model_config['max_nb_inputs']

    max_prevchars = max_outseq_len

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    tokenizer = Tokenizer()
    tokenizer.load()

    nb_chars = len(char2index)
    X1_batch = np.zeros((batch_size, max_inputseq_len, word_dims + 1), dtype=np.float32)  # слова предпосылки
    X2_batch = np.zeros((batch_size, max_inputseq_len, word_dims + 1), dtype=np.float32)  # слова вопроса
    X3_batch = np.zeros((batch_size, max_prevchars, nb_chars), dtype=np.float32)  # последние сгенерированные символы ответа

    while True:
        print('\nEnter two phrases:')
        premise = utils.console_helpers.input_kbd('premise:>').lower()
        if len(premise) == 0:
            break

        question = utils.console_helpers.input_kbd('question:>').lower()
        if len(question) == 0:
            break

        premise_words = tokenizer.tokenize(premise)
        question_words = tokenizer.tokenize(question)

        vectorize_words(lpad_words(premise_words, max_inputseq_len), X1_batch, 0, word2vec, word_dims)
        vectorize_words(lpad_words(question_words, max_inputseq_len), X2_batch, 0, word2vec, word_dims)

        answer = [BEG_CHAR]
        while True:
            for ichar, c in enumerate(rpad_chars(answer[::-1], max_prevchars)):
                X3_batch[0, ichar, char2index[c]] = 1.0

            y_pred = model.predict()
            y_pred = np.argmax(y_pred, axis=-1)
            next_char = index2char[y_pred[0]]
            if next_char == END_CHAR:
                break
            else:
                answer.append(next_char)

        answer = undress(u''.join(answer))
        print(u'answer={}'.format(answer))


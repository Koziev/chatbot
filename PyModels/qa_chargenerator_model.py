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

input_path = '../data/premise_question_answer.csv'
tmp_folder = '../tmp'
data_folder = '../data'

#NET_ARCH = 'lstm'
#NET_ARCH = 'lstm+cnn'
NET_ARCH = 'lstm(cnn)'

#GENERATOR_ARCH = 'lstm'
GENERATOR_ARCH = 'lstm(lstm)'
#GENERATOR_ARCH = 'lstm(lstm(lstm))'


#BATCH_SIZE = 1000
BATCH_SIZE = 400

# Максимальная длина ответа в символах.
MAX_ANSWER_LEN = 30

# Кол-во ядер в сверточных слоях упаковщика предложений.
nb_filters = 128

USE_WORD_MATCHING = False

initializer = 'random_normal'


# w2v_path = '~/w2v/w2v.CBOW=0_WIN=5_DIM=32.txt'
w2v_path = '~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin'
# w2v_path = '~/polygon/WordSDR2/sdr.dat'
# w2v_path = '~/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=128.txt'
# w2v_path = r'f:\Word2Vec\word_vectors_cbow=1_win=5_dim=32.txt'



# -------------------------------------------------------------------

PAD_WORD = u''
PAD_CHAR = u'\r'

# -------------------------------------------------------------------------

def prepare_answer(answer_str):
    return answer_str


# Слева добавляем пустые слова
def lpad_wordseq(words, n):
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


# Справа добавляем пустые слова
def rpad_wordseq(words, n):
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def pad_input_wordseq(words, n):
    return lpad_wordseq(words, n)


def rpad_charseq(s, n):
    return s+PAD_CHAR*max(0, n-len(s))


def vectorize_words(words, M, irow, word2vec):
    for iword, word in enumerate(words):
        if word != PAD_WORD:
            M[irow, iword, :] = word2vec[word]


def generate_rows(sequences, targets, batch_size, mode, char2id):
    batch_index = 0
    batch_count = 0

    X1_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, max_outputseq_len, output_dims), dtype=np.bool)

    while True:
        for irow, (seq, target) in enumerate(itertools.izip(sequences, targets)):
            vectorize_words(seq[0], X1_batch, batch_index, word2vec)
            vectorize_words(seq[1], X2_batch, batch_index, word2vec)
            for ichar, c in enumerate(target):
                y_batch[batch_index, ichar, char2id[c]] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield ({'input_words1': X1_batch, 'input_words2': X2_batch}, {'output': y_batch})
                else:
                    yield {'input_words1': X1_batch, 'input_words2': X2_batch}

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0

# -------------------------------------------------------------------


def decode_ystr(y, index2char):
    s = []
    for char_v in y:
        char_index = np.argmax(char_v)
        c = index2char[char_index]
        s.append(c)

    return u''.join(s).replace(PAD_CHAR, u' ').strip()



class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class VisualizeCallback(keras.callbacks.Callback):

    def __init__(self, val_input, val_output, model, weights_path, char2id):
        self.epoch = 0
        self.val_input = val_input
        self.val_output = val_output
        self.model = model
        self.weights_path = weights_path
        self.char2id = char2id
        self.id2char = dict([(i, c) for c, i in char2id.items()])
        self.best_acc = 0
        self.stop_epoch = 0
        self.early_stopping = 20
        self.wait_epoch = 0
        self.val_acc_history = []  # для сохранения кривой обучения

    def decode_str(self, y):
        return decode_ystr(y, self.id2char)

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        nval = len(self.val_input)

        # восстанавливаем сгенерированные символьные цепочки ответов, сравниваем с
        # требуемыми цепочками.
        nb_errors = 0

        # Счетчик напечатанных строк, сгенерированных моделью
        nb_shown = 0

        nb_steps = int(nval/BATCH_SIZE)

        print('')
        for step, batch in enumerate(generate_rows(self.val_input, self.val_output, BATCH_SIZE, 1, self.char2id)):
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
                wrt.write('{}\t{}\n'.format(i+1,acc))

# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Answer text generator')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--model_dir', type=str, default='../tmp')

args = parser.parse_args()

run_mode = args.run_mode
model_dir = args.model_dir


config_path = os.path.join(tmp_folder, 'qa_chargenerator.config')

tokenizer = Tokenizer()


if run_mode == 'train':
    # В этих файлах будем сохранять натренированную сетку
    arch_filepath = os.path.join(tmp_folder, 'qa_chargenerator.arch')
    weights_path = os.path.join(tmp_folder, 'qa_chargenerator.weights')

    wordchar2vector_path = os.path.join(data_folder, 'wordchar2vector.dat')
    print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    print('wc2v_dims={0}'.format(wc2v_dims))

    # --------------------------------------------------------------------------
    # Загружаем датасет, анализируем использование символов и слов.
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    print('samples.count={}'.format(df.shape[0]))

    max_inputseq_len = 0
    max_outputseq_len = 0  # максимальная длина ответа
    all_words = set([PAD_WORD])
    all_chars = set([PAD_CHAR])

    for i, record in df.iterrows():
        answer = record['answer']
        if len(answer) <= MAX_ANSWER_LEN:  # для отладки модели
            if answer not in [u'да']:
                all_chars.update(answer)
                max_outputseq_len = max(max_outputseq_len, len(answer))

                for phrase in [record['premise'], record['question']]:
                    all_chars.update(phrase)
                    words = tokenizer.tokenize(phrase)
                    all_words.update(words)
                    max_inputseq_len = max(max_inputseq_len, len(words))

    for word in wc2v.vocab:
        all_words.add(word)

    print('max_inputseq_len={}'.format(max_inputseq_len))
    print('max_outputseq_len={}'.format(max_outputseq_len))

    char2id = dict(
        [(c, i) for i, c in enumerate(itertools.chain([PAD_CHAR], filter(lambda z: z != PAD_CHAR, all_chars)))])

    nb_chars = len(all_chars)
    nb_words = len(all_words)
    print('nb_chars={}'.format(nb_chars))
    print('nb_words={}'.format(nb_words))

    # --------------------------------------------------------------------------

    print('Loading the w2v model {}'.format(os.path.expanduser(w2v_path)))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(os.path.expanduser(w2v_path), binary=not w2v_path.endswith('.txt'))
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
    # -------------------------------------------------------------------

    rnn_size = word_dims

    final_merge_size = 0  # вычисляемый параметр сетки - размер вектора на выходе энкодера

    print('Building the NN computational graph {} {}'.format(NET_ARCH, GENERATOR_ARCH))
    words_net1 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words1')
    words_net2 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words2')

    conv1 = []
    conv2 = []
    encoder_size = 0

    if NET_ARCH == 'lstm':

        # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # каждого из входных предложений. Сетка сделана общей для предпосылки и вопроса,
        # так как такое усреднение улучшает качество в сравнении с вариантом раздельных
        # сеток для каждого из предложений.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))
        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)
        encoder_size += rnn_size*2

    if NET_ARCH == 'lstm+cnn':
        # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # каждого из входных предложений. Сетка сделана общей для предпосылки и вопроса,
        # так как такое усреднение улучшает качество в сравнении с вариантом раздельных
        # сеток для каждого из предложений.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))
        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)

        encoder_size += rnn_size*2

        # добавляем входы со сверточными слоями
        for kernel_size in range(2, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            #dense2 = Dense(units=nb_filters)

            conv_layer1 = conv(words_net1)
            conv_layer1 = GlobalAveragePooling1D()(conv_layer1)
            #conv_layer1 = dense2(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = GlobalAveragePooling1D()(conv_layer2)
            #conv_layer2 = dense2(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += nb_filters

    if NET_ARCH == 'lstm(cnn)':
        for kernel_size in range(1, 4):
            # сначала идут сверточные слои, образующие детекторы словосочетаний
            # и синтаксических конструкций
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          kernel_initializer=initializer,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            lstm = recurrent.LSTM(rnn_size,
                                  return_sequences=False,
                                  kernel_initializer='random_normal')
            pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv_layer2 = lstm(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += rnn_size

    # Слой для попарной похожести слов во входных цепочках.
    if USE_WORD_MATCHING:
        pq = match(inputs=[words_net1, words_net2], axes=-1, normalize=False, match_type='dot')
        pq = Flatten()(pq)
        encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2, [pq])))
    else:
        encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2)))

    encoder_final = Dense(units=int(encoder_size), activation='relu', kernel_initializer=initializer)(encoder_merged)
    encoder_final = Dense(units=int(encoder_size), activation='relu', kernel_initializer=initializer)(encoder_final)

    # декодер генерирует цепочку символов ответа
    output_dims = nb_chars
#    decoder = Dense(units=encoder_size, activation='relu')(encoder_final)
#    decoder = Dense(units=encoder_size, activation='relu')(decoder)
#    decoder = Dense(units=encoder_size, activation='relu')(decoder)
    decoder = RepeatVector(max_outputseq_len)(encoder_final)

    # Стек из нескольких рекуррентных слоев. Предполагается, что первый
    # слой формирует грубую структуру ответа, а второй слой уточняет
    # ее до точной цепочки символов и т.д., а последний слой формирует
    # цепочку символов
    if GENERATOR_ARCH == 'lstm(lstm(lstm))':
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
    elif GENERATOR_ARCH == 'lstm(lstm)':
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)
    else:
        decoder = recurrent.LSTM(encoder_size, return_sequences=True, kernel_initializer=initializer)(decoder)

    decoder = TimeDistributed(Dense(nb_chars, activation='softmax', kernel_initializer=initializer), name='output')(decoder)

    model = Model(inputs=[words_net1, words_net2], outputs=decoder)

    #opt = 'nadam'
    #opt = 'rmsprop'
    opt = 'adam'
    #opt = keras_contrib.optimizers.FTML()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    input_data = []
    output_data = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
        premise = row['premise']
        question = row['question']
        answer = row['answer']
        if len(answer) <= MAX_ANSWER_LEN:
            if answer not in [u'да']:

                answer = prepare_answer(answer)  # эксперимент

                premise_words = pad_input_wordseq(tokenizer.tokenize(premise), max_inputseq_len)
                question_words = pad_input_wordseq(tokenizer.tokenize(question), max_inputseq_len)

                input_data.append((premise_words, question_words, premise, question))
                output_data.append(rpad_charseq(answer, max_outputseq_len))

            #if len(input_data)>=1000:
            #    break

    SEED = 123456
    TEST_SHARE = 0.2
    train_input, val_input, train_output, val_output = train_test_split(input_data,
                                                                        output_data,
                                                                        test_size=TEST_SHARE,
                                                                        random_state=SEED)

    batch_size = BATCH_SIZE

    nb_train_patterns = len(train_input)
    nb_valid_patterns = len(val_input)

    print('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    #monitor_metric = 'val_loss'
    #model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric, verbose=1, save_best_only=True, mode='auto')
    #early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    viz = VisualizeCallback(val_input, val_output, model, weights_path, char2id)

    #callbacks = [model_checkpoint, early_stopping, viz]
    callbacks = [viz]

    hist = model.fit_generator(generator=generate_rows(train_input, train_output, batch_size, 1, char2id),
                               steps_per_epoch=nb_train_patterns//batch_size,
                               epochs=1000,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(val_input, val_output, batch_size, 1, char2id),
                               validation_steps=nb_valid_patterns//batch_size
                               )

    print('Training is finished.')

    viz.save_learning_curve(os.path.join(tmp_folder, 'qa_chargenerator.learning_curve.tsv'))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'max_inputseq_len': max_inputseq_len,
                    'max_outputseq_len': max_outputseq_len,
                    'w2v_path': w2v_path,
                    'wordchar2vector_path': wordchar2vector_path,
                    'PAD_WORD': PAD_WORD,
                    'PAD_CHAR': ord(PAD_CHAR),
                    'model_folder': tmp_folder,
                    'word_dims': word_dims,
                    'arch_filepath': arch_filepath,
                    'weights_path': weights_path,
                    'char2id': char2id
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    model.load_weights(weights_path)

    nval = len(val_input)
    print(u'Финальная валидация модели на {} сэмплах'.format(nval))
    id2char = dict([(i, c) for c, i in char2id.items()])

    # Накопим кол-во ошибок и сэмплов для ответов разной длины.
    answerlen2samples = Counter()
    answerlen2errors = Counter()

    with codecs.open(os.path.join(tmp_folder, 'qa_chargenerator_model.validation.txt'), 'w', 'utf-8') as wrt:
        nb_steps = nval // BATCH_SIZE
        isample = 0
        for step, batch in enumerate(generate_rows(val_input, val_output, BATCH_SIZE, 1, char2id)):
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
                    wrt.write(u'Premise:      {}\n'.format(u' '.join(val_input[isample][0]).strip()))
                    wrt.write(u'Question:     {}\n'.format(u' '.join(val_input[isample][1]).strip()))
                    wrt.write(u'True answer:  {}\n'.format(target_chars))
                    wrt.write(u'Model answer: {}\n'.format(predicted_chars))
                    wrt.write('\n\n')
                    answerlen2errors[answer_len] += 1

                isample += 1

    # Accuracy for answers with respect to their lengths:
    with open(os.path.join(tmp_folder, 'qa_chargenerator_model.accuracy.csv'), 'w') as wrt:
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
    w2v_path = cfg['w2v_path']
    wordchar2vector_path = cfg['wordchar2vector_path']
    word_dims = cfg['word_dims']
    arch_filepath = cfg['arch_filepath']
    weights_path = cfg['weights_path']
    char2id = cfg['char2id']

    arch_filepath = os.path.join(model_dir, os.path.basename(arch_filepath))
    weights_path = os.path.join(model_dir, os.path.basename(weights_path))

    index2char = dict((i, c) for (c, i) in six.iteritems(char2id))

    print('Restoring model architecture from {}'.format(arch_filepath))
    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    print('Loading model weights from {}'.format(weights_path))
    model.load_weights(weights_path)

    print('Loading word embeddings')
    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, w2v_path)
    word_dims = embeddings.vector_size

    X1_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)
    X2_probe = np.zeros((1, max_inputseq_len, word_dims), dtype=np.float32)

    while True:
        premise = raw_input('Premise:> ').decode(sys.stdout.encoding).strip().lower()
        if len(premise) == 0:
            break

        question = raw_input('Question:> ').decode(sys.stdout.encoding).strip().lower()
        if len(question) == 0:
            break

        premise_words = pad_input_wordseq(tokenizer.tokenize(premise), max_inputseq_len)
        question_words = pad_input_wordseq(tokenizer.tokenize(question), max_inputseq_len)

        X1_probe.fill(0)
        X2_probe.fill(0)
        vectorize_words(premise_words, X1_probe, 0, embeddings)
        vectorize_words(question_words, X2_probe, 0, embeddings)

        y = model.predict([X1_probe, X2_probe])
        answer = decode_ystr(y[0], index2char)

        print(u'{}'.format(answer))

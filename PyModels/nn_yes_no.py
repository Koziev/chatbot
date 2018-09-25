# -*- coding: utf-8 -*-
'''
Тренировка модели классификации yes/no для сэмплов с несколькими (от 0 до n)
предпосылками и вопросом.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
Используется нейросетка (Keras).
Датасет "pqa_yes_no.dat" должен быть сгенерирован и находится в папке ../data (см. prepare_qa_dataset.py)
Также нужна модель встраивания слов (word2vector) и модель посимвольного встраивания (wordchar2vector).
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import sys
import gensim
import codecs
import keras.callbacks
import numpy as np
import tqdm
import argparse
import logging
import logging.handlers


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


PAD_WORD = u''
padding = 'left'


class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer



def pad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))

def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


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

    y_batch = np.zeros((batch_size, output_dims), dtype=np.bool)

    weights = np.zeros((batch_size), dtype=np.float32)
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

            if sample.answer == u'нет':
                y_batch[batch_index, 0] = True
                weights[batch_index] = 1.0  #float(nb_no+nb_yes) / nb_yes
            else:
                y_batch[batch_index, 1] = True
                weights[batch_index] = 1.0  #float(nb_no+nb_yes) / nb_no

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




# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Neural model for yes/no answer classification')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture: lstm | lstm(cnn)')
parser.add_argument('--batch_size', type=int, default=150, help='batch size for neural model training')
parser.add_argument('--input', type=str, default='../data/pqa_yes_no.dat', help='path to input dataset')
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

# настраиваем логирование в файл
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
lf = logging.FileHandler(os.path.join(tmp_folder, 'nn_yes_no.log'), mode='w')

lf.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
lf.setFormatter(formatter)
logging.getLogger('').addHandler(lf)


# В этих файлах будем сохранять натренированную сетку
config_path = os.path.join(tmp_folder, 'nn_yes_no.config')
arch_filepath = os.path.join(tmp_folder, 'nn_yes_no.arch')
weights_path = os.path.join(tmp_folder, 'nn_yes_no.weights')

if run_mode == 'train':
    logging.info('Start run_mode==train')

    max_inputseq_len = 0
    max_outputseq_len = 0 # максимальная длина ответа
    all_words = set()
    all_chars = set()

    # --------------------------------------------------------------------------

    wordchar2vector_path = os.path.join(tmp_folder,'wordchar2vector.dat')
    logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    logging.info('wc2v_dims={0}'.format(wc2v_dims))

    # --------------------------------------------------------------------------

    # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
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
                    sample = Sample(premises, question, answer)
                    samples.append(sample)

                    max_nb_premises = max(max_nb_premises, len(premises))

                    if answer == u'да':
                        nb_yes += 1
                    elif answer == u'нет':
                        nb_no += 1

                    for phrase in lines:
                        all_chars.update(phrase)
                        words = tokenizer.tokenize(phrase)
                        all_words.update(words)
                        max_inputseq_len = max(max_inputseq_len, len(words))

                    lines = []

            else:
                lines.append(line)

    logging.info('samples.count={}'.format(len(samples)))
    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_nb_premises={}'.format(max_nb_premises))

    for word in wc2v.vocab:
        all_words.add(word)
        all_chars.update(word)

    word2id = dict([(c, i) for i, c in enumerate(itertools.chain([PAD_WORD], filter(lambda z: z != PAD_WORD,all_words)))])

    nb_chars = len(all_chars)
    nb_words = len(all_words)
    logging.info('nb_chars={}'.format(nb_chars))
    logging.info('nb_words={}'.format(nb_words))

    logging.info('nb_yes={}'.format(nb_yes))
    logging.info('nb_no={}'.format(nb_no))

    # --------------------------------------------------------------------------

    logging.info(u'Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    logging.info('w2v_dims={0}'.format(w2v_dims))

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
                    'max_nb_premises': max_nb_premises,
                    'arch_filepath': arch_filepath,
                    'weights_filepath': weights_path
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)
    # -------------------------------------------------------------------

    logging.info('Constructing neural net: {}...'.format(net_arch))

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

    # Тренируем модель, которая будет выдавать ответы yes или no.
    output_dims = 2

    decoder = Dense(encoder_size//2, activation='relu')(encoder_final)
    #decoder = Dense(encoder_size//3, activation='relu')(decoder)
    decoder = Dense(encoder_size//4, activation='relu')(decoder)
    #decoder = BatchNormalization()(decoder)
    decoder = Dense(output_dims, activation='softmax', name='output')(decoder)

    model = Model(inputs=inputs, outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
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
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=20, verbose=1, mode='auto')
    callbacks = [model_checkpoint, early_stopping]

    hist = model.fit_generator(generator=generate_rows(max_nb_premises, train_samples, batch_size, 1),
                               steps_per_epoch=nb_train_patterns//batch_size,
                               epochs=100,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(max_nb_premises, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns//batch_size)
    logging.info('max val_acc={}'.format(max(hist.history['val_acc'])))

    # Загрузим лучшие веса и прогоним валидационные паттерны через модель,
    # чтобы получить f1 score.
    model.load_weights(weights_path)

    for v in generate_rows(max_nb_premises, val_samples, nb_valid_patterns, 1):
        x = v[0]
        y_val = v[1]['output']
        break

    y_pred = model.predict(x)[:, 1]
    y_pred = (y_pred >= 0.5).astype(np.int)
    f1 = sklearn.metrics.f1_score(y_true=y_val[:, 1], y_pred=y_pred)
    print('val f1={}'.format(f1))

    # Сохраним в текстовом файле для визуальной проверки результаты валидации по всем сэмплам
    for v in generate_rows(max_nb_premises, samples, len(samples), 1):
        x = v[0]
        break

    y_pred = model.predict(x)[:, 1]
    y_pred = (y_pred >= 0.5).astype(np.int)

    with codecs.open(os.path.join(tmp_folder, 'nn_yes_no.validation.txt'), 'w', 'utf-8') as wrt:
        for isample, sample in enumerate(samples):
            if isample > 0:
                wrt.write('\n\n')

            for premise in sample.premises:
                wrt.write(u'P: {}\n'.format(premise))
            wrt.write(u'Q: {}\n'.format(sample.question))
            wrt.write(u'A: {}\n'.format(sample.answer))

            pred = u'да' if y_pred[isample] else u'нет'
            wrt.write(u'model: {}\n'.format(pred))

            #if pred == u'нет':
            #    for ipremise in range(max_nb_premises):
            #        wrt.write('\nX[{}]={}\n'.format(ipremise, x['premise{}'.format(ipremise)][isample]))
            #    wrt.write('\n')


if run_mode == 'query':
    # Ручное консольное тестирование модели, натренированной ранее с помощью --run_mode train

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        max_inputseq_len = model_config['max_inputseq_len']
        w2v_path = model_config['w2v_path']
        wordchar2vector_path = model_config['wordchar2vector_path']
        word_dims = model_config['word_dims']
        max_nb_premises = model_config['max_nb_premises']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    print('wc2v_dims={0}'.format(wc2v_dims))

    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))

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
            vectorize_words(words, Xn_probe[ipremise], 0, word2vec)

        words = tokenizer.tokenize(question)
        words = pad_wordseq(words, max_inputseq_len)
        vectorize_words(words, Xn_probe[max_nb_premises], 0, word2vec)

        y_probe = model.predict(x=inputs)

        print('p(no) ={}'.format(y_probe[0][0]))
        print('p(yes)={}'.format(y_probe[0][1]))

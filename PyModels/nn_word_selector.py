# -*- coding: utf-8 -*-
"""
Тренировка модели селектора слов - выбор слов из предпосылки и вопросы, которые будут
использоваться для генерации ответа. Задача модели - отобрать слова, на основе которых
генеративная грамматика будет строить ответ.

Для вопросно-ответной системы https://github.com/Koziev/chatbot.

Используется нейросетка (Keras).

Датасет "pqa_all.dat" должен быть сгенерирован и находится в папке ../data (см. prepare_qa_dataset.py).
Также нужна модель встраивания слов (word2vector) и модель посимвольного встраивания (wordchar2vector).
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import gensim
import io
import math
import numpy as np
import argparse
import logging

import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sklearn.metrics

import rupostagger
import rulemma

from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers


PAD_WORD = u''
padding = 'left'



# TODO - вынести тезаурус в отдельную библиотеку
class Thesaurus:
    def __init__(self):
        self.word2links = dict()

    def load(self, thesaurus_path):
        logging.info(u'Loading thesaurus from "{}"'.format(thesaurus_path))
        with io.open(thesaurus_path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('|')
                if len(tx) == 5:
                    word1 = tx[0].replace(u' - ', u'-').lower()
                    pos1 = tx[1]
                    word2 = tx[2].replace(u' - ', u'-').lower()
                    pos2 = tx[3]
                    relat = tx[4]

                    #if word1 == u'быть' or word2 == u'быть':
                    #    continue

                    if word1 != word2 and word1:  # in all_words and word2 in all_words:
                        if word1 not in self.word2links:
                            self.word2links[word1] = []
                        self.word2links[word1].append((word2, pos2, relat))

        self.word2links[u'ты'] = [(u'твой', u'ПРИЛАГАТЕЛЬНОЕ', u'в_прил')]
        self.word2links[u'я'] = [(u'мой', u'ПРИЛАГАТЕЛЬНОЕ', u'в_прил')]
        self.word2links[u'мы'] = [(u'наш', u'ПРИЛАГАТЕЛЬНОЕ', u'в_прил')]
        self.word2links[u'вы'] = [(u'ваш', u'ПРИЛАГАТЕЛЬНОЕ', u'в_прил')]
        logging.info('{} items in thesaurus loaded'.format(len(self.word2links)))

    def get_linked(self, word1):
        res = []
        if word1 in self.word2links:
            for link in self.word2links[word1]:
                res.append(link)
        return res

    def are_linked(self, lemma1, lemma2):
        if lemma1 in self.word2links:
            if lemma2 in self.word2links:
                for link in self.word2links[lemma2]:
                    if link[0] == lemma1:
                        if link[2] not in (u'в_класс', u'член_класса'):
                            return True

        return False


def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


class Sample:
    def __init__(self, premises, question, answer, word, label):
        assert(label in (0, 1))
        assert(len(answer) > 0)
        assert(len(word) > 0)
        self.premises = premises[:]
        self.question = question
        self.answer = answer
        self.word = word
        self.label = label


def pad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def load_embeddings(tmp_folder, word2vector_path, computed_params):
    wordchar2vector_path = os.path.join(tmp_folder, 'wordchar2vector.dat')
    logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
    wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
    wc2v_dims = len(wc2v.syn0[0])
    logging.info('wc2v_dims={0}'.format(wc2v_dims))

    logging.info(u'Loading the w2v model {}'.format(word2vector_path))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
    w2v_dims = len(w2v.syn0[0])
    logging.info('w2v_dims={0}'.format(w2v_dims))

    word_dims = w2v_dims + wc2v_dims
    computed_params['word_dims'] = word_dims

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

    return word2vec, word_dims, wordchar2vector_path


def load_samples(input_path, tokenizer, tagger, lemmatizer, thesaurus):
    # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
    logging.info(u'Loading samples from {}'.format(input_path))
    samples = []
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    max_inputseq_len = 0  # макс. длина предпосылок и вопроса в словах

    with io.open(input_path, 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:

            # начало отладки
            if len(samples) > 1000000:
                print('DEBUG@175 - !!! dataset truncation !!!!')
                break
            # конец отладки

            line = line.strip()
            if len(line) == 0:
                if len(lines) > 0:
                    premises = lines[:-2]
                    question = lines[-2]
                    answer = lines[-1]

                    if answer not in (u'да', u'нет'):
                        max_nb_premises = max(max_nb_premises, len(premises))
                        for phrase in lines:
                            words = tokenizer.tokenize(phrase)
                            max_inputseq_len = max(max_inputseq_len, len(words))

                        # Теперь проверяем каждое слово в предпосылках и вопросе, участвует ли оно (одна из
                        # словарных форм) в ответе.
                        a_tokens = []
                        a_lemmas = set()
                        tokens = tokenizer.tokenize(answer)
                        tags = tagger.tag(tokens)
                        lemmas = lemmatizer.lemmatize(tags)
                        for token_info in lemmas:
                            word = token_info[0]
                            lemma = token_info[2]
                            a_tokens.append((word, lemma))
                            a_lemmas.add(lemma)

                        probed_words = set()
                        for phrase in itertools.chain(premises, [question]):
                            tokens = tokenizer.tokenize(phrase)
                            tags = tagger.tag(tokens)
                            lemmas = lemmatizer.lemmatize(tags)
                            for token_info in lemmas:
                                word = token_info[0].lower()
                                if word not in probed_words:
                                    probed_words.add(word)
                                    lemma = token_info[2]

                                    label = 0
                                    if lemma in a_lemmas:
                                        label = 1
                                    else:
                                        # Ищем однокоренное слово через тезаурус
                                        for answer_lemma in a_lemmas:
                                            if thesaurus.are_linked(lemma, answer_lemma):
                                                label = 1
                                                break

                                    sample = Sample(premises, question, answer, word, label)
                                    samples.append(sample)
                                    #print('DEBUG@226 len(samples)={}'.format(len(samples)))

                    lines = []
            else:
                lines.append(line)

    logging.info('samples.count={}'.format(len(samples)))
    logging.info('max_inputseq_len={}'.format(max_inputseq_len))
    logging.info('max_nb_premises={}'.format(max_nb_premises))

    nb_0 = sum((sample.label==0) for sample in samples)
    nb_1 = sum((sample.label==1) for sample in samples)

    logging.info('nb_0={}'.format(nb_0))
    logging.info('nb_1={}'.format(nb_1))

    computed_params = {'max_nb_premises': max_nb_premises,
                       'max_inputseq_len': max_inputseq_len,
                       'nb_1': nb_1,
                       'nb_0': nb_0
                       }

    return samples, computed_params


def vectorize_words(words, M, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def create_model(params, computed_params):
    net_arch = params['net_arch']
    logging.info('Constructing neural net: {}...'.format(net_arch))

    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    max_nb_premises = computed_params['max_nb_premises']

    inputs = []
    input_question = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='question')
    inputs.append(input_question)

    for ipremise in range(max_nb_premises):
        input_premise = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='premise{}'.format(ipremise))
        inputs.append(input_premise)

    input_word = Input(shape=(word_dims,), dtype='float32', name='word')

    layers = []
    encoder_size = 0

    if net_arch == 'lstm':
        rnn_size = params['rnn_size']

        # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # предложения. Этот слой общий для всех входных предложений.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                        input_shape=(max_inputseq_len, word_dims),
                                                        return_sequences=False))

        for input in inputs:
            encoder_rnn = shared_words_rnn(input)
            layers.append(encoder_rnn)
            encoder_size += rnn_size*2
    elif net_arch == 'lstm(cnn)':
        rnn_size = params['rnn_size']
        nb_filters = params['nb_filters']
        max_kernel_size = params['max_kernel_size']

        for kernel_size in range(1, max_kernel_size+1):
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

                if params['pooling'] == 'max':
                    pooling = keras.layers.MaxPooling1D()
                elif params['pooling'] == 'average':
                    pooling = keras.layers.AveragePooling1D()
                else:
                    raise NotImplementedError()

                conv_layer1 = pooling(conv_layer1)

                conv_layer1 = lstm(conv_layer1)
                layers.append(conv_layer1)
                encoder_size += rnn_size
    elif net_arch == 'cnn':
        nb_filters = params['nb_filters']
        max_kernel_size = params['max_kernel_size']

        for kernel_size in range(1, max_kernel_size+1):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            for input in inputs:
                conv_layer1 = conv(input)

                if params['pooling'] == 'max':
                    pooling = keras.layers.GlobalMaxPooling1D()
                elif params['pooling'] == 'average':
                    pooling = keras.layers.GlobalAveragePooling1D()
                else:
                    raise NotImplementedError()

                conv_layer1 = pooling(conv_layer1)
                layers.append(conv_layer1)
    else:
        raise NotImplementedError()

    layers.append(input_word)

    encoder_merged = keras.layers.concatenate(inputs=list(layers))
    decoder = encoder_merged

    if params['units1'] > 0:
        decoder = Dense(params['units1'], activation='relu')(decoder)

        if params['units2'] > 0:
            decoder = Dense(params['units2'], activation='relu')(decoder)

            if params['units3'] > 0:
                decoder = Dense(params['units3'], activation='relu')(decoder)

    output_dims = 2
    decoder = Dense(output_dims, activation='softmax', name='output')(decoder)

    inputs2 = list(itertools.chain(inputs, [input_word]))
    model = Model(inputs=inputs2, outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    #model.summary()

    return model


def generate_rows(params, computed_params, samples, batch_size, mode):
    batch_index = 0
    batch_count = 0
    max_inputseq_len = computed_params['max_inputseq_len']
    word_dims = computed_params['word_dims']
    nb_premises = computed_params['max_nb_premises']
    w1_weight = params['w1_weight']

    Xn_batch = []
    for _ in range(nb_premises + 1):
        x = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_batch.append(x)

    inputs = {}
    for ipremise in range(nb_premises):
        inputs['premise{}'.format(ipremise)] = Xn_batch[ipremise]
    inputs['question'] = Xn_batch[nb_premises]

    X_word = np.zeros((batch_size, word_dims), dtype=np.float32)
    inputs['word'] = X_word

    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    weights = np.zeros((batch_size), dtype=np.float32)
    weights.fill(1.0)

    missing_words = set()

    while True:
        for irow, sample in enumerate(samples):
            for ipremise, premise in enumerate(sample.premises):
                words = tokenizer.tokenize(premise)
                words = pad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[ipremise], batch_index, word2vec)

            words = tokenizer.tokenize(sample.question)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_batch[nb_premises], batch_index, word2vec)

            # проверяемое слово
            if sample.word not in word2vec:
                if sample.word not in missing_words:
                    logging.error(u'Word \"{}\" missing in word2vec'.format(sample.word))
                    missing_words.add(sample.word)
            else:
                X_word[batch_index, :] = word2vec[sample.word]

            if sample.label == 0:
                y_batch[batch_index, 0] = True
                weights[batch_index] = 1.0
            else:
                y_batch[batch_index, 1] = True
                weights[batch_index] = w1_weight

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield (inputs, {'output': y_batch}, weights)
                else:
                    yield inputs

                # очищаем матрицы порции для новой порции
                X_word.fill(0)
                for x in Xn_batch:
                    x.fill(0)
                y_batch.fill(0)
                batch_index = 0


def train_model(model, params, computed_params, train_samples, val_samples):
    nb_train_patterns = len(train_samples)
    nb_valid_patterns = len(val_samples)
    logging.info('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_acc'
    model_checkpoint = ModelCheckpoint(weights_path,
                                       monitor=monitor_metric,
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto')
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=5, verbose=1, mode='auto')
    callbacks = [model_checkpoint, early_stopping]

    batch_size = params['batch_size']
    hist = model.fit_generator(generator=generate_rows(params, computed_params, train_samples, batch_size, 1),
                               steps_per_epoch=nb_train_patterns // batch_size,
                               epochs=1000,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows(params, computed_params, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns // batch_size)
    logging.info('max val_acc={}'.format(max(hist.history['val_acc'])))
    model.load_weights(weights_path)


def score_model(model, params, computed_params, val_samples):
    # прогоним валидационные паттерны через модель, чтобы получить f1 score.
    nb_valid_patterns = len(val_samples)
    for v in generate_rows(params, computed_params, val_samples, nb_valid_patterns, 1):
        x = v[0]
        y_val = v[1]['output']
        break

    n0 = sum(y_val[:, 1] == 0)
    n1 = sum(y_val[:, 1] == 1)

    logging.info('targets: n0={} n1={}'.format(n0, n1))

    y_pred0 = model.predict(x)[:, 1]
    y_pred = (y_pred0 >= 0.5).astype(np.int)
    f1 = sklearn.metrics.f1_score(y_true=y_val[:, 1], y_pred=y_pred)
    logging.info('val f1={}'.format(f1))

    #score = -sklearn.metrics.log_loss(y_true=y_val[:, 1], y_score=y_pred0)
    score = sklearn.metrics.roc_auc_score(y_true=y_val[:, 1], y_score=y_pred0)
    logging.info('roc_auc_score={}'.format(score))

    return f1, score


def report_model(model, params, computed_params, samples):
    # Сохраним в текстовом файле для визуальной проверки результаты валидации по всем сэмплам
    for v in generate_rows(params, computed_params, samples, len(samples), 1):
        x = v[0]
        break

    y_pred0 = model.predict(x)[:, 1]
    y_pred = (y_pred0 >= 0.5).astype(np.int)

    with io.open(os.path.join(tmp_folder, 'nn_word_selector.validation.txt'), 'w', encoding='utf-8') as wrt:
        for isample, sample in enumerate(samples):
            if isample > 0:
                wrt.write(u'\n\n')

            for premise in sample.premises:
                wrt.write(u'P: {}\n'.format(premise))
            wrt.write(u'Q: {}\n'.format(sample.question))
            wrt.write(u'A: {}\n'.format(sample.answer))
            wrt.write(u'word={} true_label={} predicted_label={} (y={})\n'.format(sample.word, sample.label, y_pred[isample], y_pred0[isample]))

# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Neural model for word selector')
parser.add_argument('--run_mode', type=str, default='train', choices='train gridsearch query'.split(), help='what to do: train | query | gridsearch')
parser.add_argument('--batch_size', type=int, default=250, help='batch size for neural model training')
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
run_mode = args.run_mode

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_word_selector.log'))

# В этих файлах будем сохранять натренированную сетку
config_path = os.path.join(tmp_folder, 'nn_word_selector.config')
arch_filepath = os.path.join(tmp_folder, 'nn_word_selector.arch')
weights_path = os.path.join(tmp_folder, 'nn_word_selector.weights')

thesaurus = Thesaurus()
thesaurus.load(os.path.join(data_folder, 'dict/links.csv'))

if run_mode == 'gridsearch':
    logging.info('Start gridsearch')

    tokenizer = Tokenizer()
    tokenizer.load()

    tagger = rupostagger.RuPosTagger()
    tagger.load()

    lemmatizer = rulemma.Lemmatizer()
    lemmatizer.load()

    samples, computed_params = load_samples(input_path, tokenizer, tagger, lemmatizer, thesaurus)

    word2vec, word_dims, wordchar2vector_path = load_embeddings(tmp_folder, word2vector_path, computed_params)

    best_params = None
    best_score = -np.inf

    n0 = computed_params['nb_no']
    n1 = computed_params['nb_yes']

    params = dict()
    crossval_count = 0
    for net_arch in ['cnn']:  #  'lstm' 'cnn' 'lstm(cnn)'
        params['net_arch'] = net_arch

        for w1_weight in [(n0/float(n0+n1)), math.sqrt((n0/float(n0+n1))), 1.0]:
            params['w1_weight'] = w1_weight

            for rnn_size in [32, 48] if net_arch in ['lstm', 'lstm(cnn)'] else [0]:
                params['rnn_size'] = rnn_size

                for nb_filters in [160, 180] if net_arch in ['cnn', 'lstm(cnn)'] else [0]:
                    params['nb_filters'] = nb_filters

                    for min_kernel_size in [1]:
                        params['min_kernel_size'] = min_kernel_size

                        for max_kernel_size in [3] if net_arch in ['cnn', 'lstm(cnn)'] else [0]:
                            params['max_kernel_size'] = max_kernel_size

                            for pooling in ['max'] if net_arch in ['cnn', 'lstm(cnn)'] else ['']:  # , 'average'
                                params['pooling'] = pooling

                                for units1 in [32]:
                                    params['units1'] = units1

                                    for units2 in [0]:
                                        params['units2'] = units2

                                        for units3 in [0]:
                                            params['units3'] = units3

                                            for batch_size in [80, 100, 120]:
                                                params['batch_size'] = batch_size

                                                for optimizer in ['nadam']:
                                                    params['optimizer'] = optimizer

                                                    crossval_count += 1
                                                    logging.info('Crossvalidation #{} for {}'.format(crossval_count, get_params_str(params)))

                                                    kf = KFold(n_splits=3)
                                                    scores = []
                                                    for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
                                                        logging.info('KFold[{}]'.format(ifold))
                                                        train_samples = [samples[i] for i in train_index]
                                                        val12_samples = [samples[i] for i in val_index]

                                                        SEED = 123456
                                                        TEST_SHARE = 0.2
                                                        val_samples, finval_samples = train_test_split(val12_samples, test_size=0.5,
                                                                                                       random_state=SEED)

                                                        model = create_model(params, computed_params)
                                                        train_model(model, params, computed_params, train_samples, val_samples)

                                                        f1_score, score = score_model(model, params, computed_params, finval_samples)
                                                        scores.append(score)

                                                    score = np.mean(scores)
                                                    score_std = np.std(scores)
                                                    logging.info('Crossvalidation #{} score={} std={}'.format(crossval_count, score, score_std))
                                                    if score > best_score:
                                                        best_params = params.copy()
                                                        best_score = score
                                                        logging.info('!!! NEW BEST score={} params={}'.format(best_score, get_params_str(best_params)))

    logging.info('Grid search complete, best_score={} best_params={}'.format(best_score, get_params_str(best_params)))


if run_mode == 'train':
    logging.info('Start run_mode==train')

    tokenizer = Tokenizer()
    tokenizer.load()

    logging.info('Loading tagger...')
    tagger = rupostagger.RuPosTagger()
    tagger.load()

    logging.info('Loading lemmatizer...')
    lemmatizer = rulemma.Lemmatizer()
    lemmatizer.load()

    samples, computed_params = load_samples(input_path, tokenizer, tagger, lemmatizer, thesaurus)

    word2vec, word_dims, wordchar2vector_path = load_embeddings(tmp_folder, word2vector_path, computed_params)

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'engine': 'nn',
                    'max_inputseq_len': computed_params['max_inputseq_len'],
                    'max_nb_premises': computed_params['max_nb_premises'],
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
        json.dump(model_config, f, indent=4)

    params = dict()
    params['net_arch'] = 'lstm'  #''lstm(cnn)'
    params['rnn_size'] = 150  #64
    params['w1_weight'] = 1.0
    params['nb_filters'] = 150
    params['min_kernel_size'] = 1
    params['max_kernel_size'] = 3
    params['pooling'] = 'max'
    params['units1'] = 80
    params['units2'] = 0  #64
    params['units3'] = 0  #32
    params['batch_size'] = 200  # 250
    params['optimizer'] = 'nadam'

    model = create_model(params, computed_params)

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=123456)
    train_model(model, params, computed_params, train_samples, val_samples)
    f1_score, logloss = score_model(model, params, computed_params, val_samples)
    report_model(model, params, computed_params, samples)


if run_mode == 'query':
    logging.info('Start run_mode==query')

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

    computed_params = dict()
    word2vec, word_dims, wordchar2vector_path = load_embeddings(tmp_folder, w2v_path, computed_params)

    tokenizer = Tokenizer()
    tokenizer.load()

    #logging.info('Loading tagger...')
    #tagger = rupostagger.RuPosTagger()
    #tagger.load()

    #logging.info('Loading lemmatizer...')
    #lemmatizer = rulemma.Lemmatizer()
    #lemmatizer.load()

    Xn_probe = []
    for i in range(max_nb_premises + 1):
        Xn_probe.append(np.zeros((1, max_inputseq_len, word_dims), dtype='float32'))
    X_word = np.zeros((1, word_dims), dtype='float32')

    while True:
        print('\nEnter 0 to {} premises and one question:'.format(max_nb_premises))
        premises = []
        question = None
        for ipremise in range(max_nb_premises):
            premise = utils.console_helpers.input_kbd('premise #{} :> '.format(ipremise)).strip().lower()
            if len(premise) == 0:
                break
            if premise[-1] == u'?':
                question = premise
                break

            premises.append(premise)

        if question is None:
            question = utils.console_helpers.input_kbd('question:> ').strip().lower()
        if len(question) == 0:
            break

        # Очистим входные тензоры перед заполнением новыми данными
        for X in Xn_probe:
            X.fill(0)

        # Векторизуем входные данные - предпосылки и вопрос
        for ipremise, premise in enumerate(premises):
            words = tokenizer.tokenize(premise)
            words = pad_wordseq(words, max_inputseq_len)
            vectorize_words(words, Xn_probe[ipremise], 0, word2vec)

        question = question.replace(u'?', u'').strip()
        words = tokenizer.tokenize(question)
        words = pad_wordseq(words, max_inputseq_len)
        vectorize_words(words, Xn_probe[max_nb_premises], 0, word2vec)

        # Идем по всем словам в предпосылках и в вопросе, получаем вероятность
        # их участия в генерации.
        input_words = set()
        for phrase in itertools.chain(premises, [question]):
            words = tokenizer.tokenize(phrase.lower())
            input_words.update(words)

        word_p = []
        for probe_word in input_words:
            X_word.fill(0)

            # Вход для проверяемого слова:
            X_word[0, :] = word2vec[probe_word]

            inputs = dict()
            for ipremise in range(max_nb_premises):
                inputs['premise{}'.format(ipremise)] = Xn_probe[ipremise]
            inputs['question'] = Xn_probe[max_nb_premises]
            inputs['word'] = X_word

            y_probe = model.predict(x=inputs)
            p = y_probe[0][1]
            word_p.append((probe_word, p))

        for word, p in sorted(word_p, key=lambda z: -z[1]):
            print(u'{:15s}\t{}'.format(word, p))

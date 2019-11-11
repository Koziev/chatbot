# -*- coding: utf-8 -*-
"""
Модель для определения СИНОНИМИИ - семантической эквивалентности двух фраз,
включая позиционные и синтаксические перефразировки, лексические и фразовые
синонимы. В отличие от модели для РЕЛЕВАНТНОСТИ предпосылки и вопроса, в этой
модели предполагается, что объем информации в обеих фразах примерно одинаков,
то есть "кошка спит" и "черная кошка сладко спит" не считаются полными синонимами.

Вариант архитектуры с TRIPLE LOSS.

10.03.2019 Добавлен вариант с представлением текста через SentencePiece
11.03.2019 Добавлен отдельный режим выполнения --run_mode gridsearch для подбора гиперпараметров
27.10.2019 Метрики для валидации переделаны на precision@1 и mean reciprocal rank

Для проекта чатбота https://github.com/Koziev/chatbot
"""

from __future__ import division
from __future__ import print_function

import codecs
import itertools
import json
import os
import io
import argparse
import random
import logging
import operator
import numpy as np
import pandas as pd
import yaml

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.models import Model
from keras.models import model_from_json
import keras.regularizers

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import sentencepiece as spm

import matplotlib.pyplot as plt

from ruchatbot.utils.tokenizer import Tokenizer
from ruchatbot.utils.padding_utils import lpad_wordseq, rpad_wordseq
from ruchatbot.utils.padding_utils import PAD_WORD
from trainers.word_embeddings import WordEmbeddings
import ruchatbot.utils.console_helpers
import ruchatbot.utils.logging_helpers


# Длина вектора предложения
PHRASE_DIM = 1000

# Предложения приводятся к единой длине путем добавления слева или справа
# пустых токенов. Параметр padding определяет режим выравнивания.
padding = 'right'


# Кол-во фолдов в кроссвалидации
CVFOLDS = 5


PAD_TOKEN = u''


random.seed(123456789)
np.random.seed(123456789)


class Sample3:
    def __init__(self, anchor, positive, negative):
        assert(len(anchor) > 0)
        assert(len(positive) > 0)
        self.anchor = anchor
        self.positive = positive
        self.negative = negative


def select_items(items, indeces):
    return [items[i] for i in indeces]


def ngrams(s, n):
    #return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2)) / float(1e-8 + len(shingles1 | shingles2))


def select_most_similar(phrase1, phrases2, topn):
    sims = [(phrase2, jaccard(phrase1, phrase2, 3)) for phrase2 in phrases2]
    sims = sorted(sims, key=lambda z: -z[1])
    return list(map(operator.itemgetter(0), sims[:topn]))


def collect_strings(d):
    res = []

    if isinstance(d, unicode):
        res.append(d)
    elif isinstance(d, list):
        for item in d:
            res.extend(collect_strings(item))
    elif isinstance(d, dict):
        for k, node in d.items():
            res.extend(collect_strings(node))

    return res


def load_strings_from_yaml(yaml_path):
    res = []
    with io.open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        strings = collect_strings(data)
        for phrase in strings:
            phrase = phrase.strip()
            if u'_' not in phrase and any((c in u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя') for c in phrase):
                res.append(phrase)
    return res


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


def vectorize_words(words, X_batch, irow, word2vec):
    for iword, word in enumerate(words):
        if word != PAD_WORD:
            X_batch[irow, iword, :] = word2vec[word]


def generate_rows_words(params, computed_params, samples, mode):
    if mode == 1:
        # При обучении сетки каждую эпоху тасуем сэмплы.
        random.shuffle(samples)

    batch_index = 0
    batch_count = 0

    batch_size = params['batch_size']
    max_wordseq_len = computed_params['max_wordseq_len']
    word_dims = computed_params['word_dims']
    w2v = computed_params['embeddings']

    X1_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)
    X3_batch = np.zeros((batch_size, max_wordseq_len, word_dims), dtype=np.float32)

    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    while True:
        for irow, sample in enumerate(samples):
            vectorize_words(sample.anchor_words, X1_batch, batch_index, w2v)
            vectorize_words(sample.positive_words, X2_batch, batch_index, w2v)
            vectorize_words(sample.negative_words, X3_batch, batch_index, w2v)

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                xx = {'input_anchor': X1_batch,
                      'input_positive': X2_batch,
                      'input_negative': X3_batch}
                if mode == 1:
                    yield (xx, {'output': y_batch})
                else:
                    yield xx

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                X3_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


def generate_rows_pieces(params, computed_params, samples, mode):
    if mode == 1:
        # При обучении сетки каждую эпоху тасуем сэмплы.
        random.shuffle(samples)

    batch_index = 0
    batch_count = 0

    batch_size = params['batch_size']
    max_wordseq_len = computed_params['max_wordseq_len']

    X1_batch = np.zeros((batch_size, max_wordseq_len,), dtype=np.int32)
    X2_batch = np.zeros((batch_size, max_wordseq_len,), dtype=np.int32)
    X3_batch = np.zeros((batch_size, max_wordseq_len,), dtype=np.int32)

    y_batch = np.zeros((batch_size, 2), dtype=np.bool)

    while True:
        for irow, sample in enumerate(samples):
            X1_batch[batch_index, :] = sample.anchor_words
            X2_batch[batch_index, :] = sample.positive_words
            X3_batch[batch_index, :] = sample.negative_words

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                xx = {'input_anchor': X1_batch,
                      'input_positive': X2_batch,
                      'input_negative': X3_batch}
                if mode == 1:
                    yield (xx, {'output': y_batch})
                else:
                    yield xx

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                X3_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


def triplet_loss(y_true, y_pred):
    """
    See also https://stackoverflow.com/questions/38260113/implementing-contrastive-loss-and-triplet-loss-in-tensorflow

    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    alpha = 2.00  #0.5

    anchor = y_pred[:, 0:PHRASE_DIM]
    positive = y_pred[:, PHRASE_DIM:2 * PHRASE_DIM]
    negative = y_pred[:, 2 * PHRASE_DIM:3 * PHRASE_DIM]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    loss = K.mean(loss, axis=-1)
    return loss


def triplet_loss2(y_true, y_pred):
    """Модификация triplet_loss с cos-метрикой"""
    alpha = 0.5

    anchor = y_pred[:, 0:PHRASE_DIM]
    positive = y_pred[:, PHRASE_DIM:2 * PHRASE_DIM]
    negative = y_pred[:, 2 * PHRASE_DIM:3 * PHRASE_DIM]

    # distance between the anchor and the positive
    pos_dist = -K.sum(anchor * positive, axis=-1)

    # distance between the anchor and the negative
    neg_dist = -K.sum(anchor * negative, axis=-1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    loss = K.mean(loss, axis=-1)
    return loss


def lossless_triplet_loss(y_true, y_pred, N=PHRASE_DIM, beta=PHRASE_DIM, epsilon=1e-8):
    """
    Implementation of the triplet loss function
    https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)


    Returns:
    loss -- real number, value of the loss
    """

    anchor = y_pred[:, 0:PHRASE_DIM]
    positive = y_pred[:, PHRASE_DIM:2 * PHRASE_DIM]
    negative = y_pred[:, 2 * PHRASE_DIM:3 * PHRASE_DIM]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # Non Linear Values

    # -ln(-x/N+1)
    pos_dist = -K.log(-pos_dist / beta + 1 + epsilon)
    neg_dist = -K.log(-(N - neg_dist) / beta + 1 + epsilon)

    # compute loss
    loss = neg_dist + pos_dist

    return loss


# -------------------------------------------------------------------

def create_model(params, computed_params):
    global PHRASE_DIM

    PHRASE_DIM = params['phrase_dim']

    max_wordseq_len = computed_params['max_wordseq_len']
    net_arch = params['net_arch']

    logging.info('Constructing the NN model arch={}...'.format(net_arch))

    if params['repres'] == 'words':
        word_dims = computed_params['word_dims']
        input_anchor = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_anchor')
        input_positive = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_positive')
        input_negative = Input(shape=(max_wordseq_len, word_dims,), dtype='float32', name='input_negative')

        anchor_flow = input_anchor
        positive_flow = input_positive
        negative_flow = input_negative
    elif params['repres'] == 'pieces':
        input_anchor = Input(shape=(max_wordseq_len,), dtype='int32', name='input_anchor')
        input_positive = Input(shape=(max_wordseq_len,), dtype='int32', name='input_positive')
        input_negative = Input(shape=(max_wordseq_len,), dtype='int32', name='input_negative')

        emb = keras.layers.Embedding(len(computed_params['token2index']),
                                     params['token_dim'],
                                     input_length=computed_params['max_wordseq_len'],
                                     trainable=True,
                                     mask_zero=True)

        anchor_flow = emb(input_anchor)
        positive_flow = emb(input_positive)
        negative_flow = emb(input_negative)
    else:
        raise NotImplementedError()

    if net_arch == 'lstm':
        a_regul = None
        if params['l1'] > 0.0:
            a_regul = keras.regularizers.l1(params['l1'])

        lstm = recurrent.LSTM(params['phrase_dim'],
                              return_sequences=False,
                              activity_regularizer=a_regul,
                              name='anchor_flow')

        anchor_flow = lstm(anchor_flow)
        positive_flow = lstm(positive_flow)
        negative_flow = lstm(negative_flow)

    output = keras.layers.concatenate(inputs=[anchor_flow, positive_flow, negative_flow], name='output')

    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=output)
    model.compile(loss=triplet_loss, optimizer=params['optimizer'])
    #model.summary()

    #keras.utils.plot_model(model,
    #                       to_file=os.path.join(tmp_folder, 'nn_synonymy_tripletloss.arch.png'),
    #                       show_shapes=False,
    #                       show_layer_names=True)

    return model, input_anchor, anchor_flow


def get_params_str(params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in params.items())


def spm2tokens(splitter, text, max_text_len, token2index):
    tokens = splitter.EncodeAsPieces(text)
    lt = len(tokens)
    tail = max_text_len - lt
    return np.array([token2index.get(t, 0) for t in tokens] + [0] * tail)


def prepare_data(input_path, params, max_samples):
    logging.info('prepare_data for "%s', get_params_str(params))
    samples3 = []
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    for anchor, positive, negative in zip(df['anchor'].values, df['positive'].values, df['negative'].values):
        samples3.append(Sample3(anchor, positive, negative))

    if len(samples3) > max_samples:
        samples3 = random.sample(samples3, max_samples)

    computed_params = dict()

    if params['repres'] == 'words':
        embeddings = WordEmbeddings.load_word_vectors(params['wordchar2vector_path'], params['word2vector_path'])
        computed_params['embeddings'] = embeddings
        computed_params['word_dims'] = embeddings.vector_size

        tokenizer = Tokenizer()
        tokenizer.load()
        computed_params['tokenizer'] = tokenizer

        max_wordseq_len = 0
        for sample in samples3:
            for phrase in [sample.anchor, sample.positive, sample.negative]:
                words = tokenizer.tokenize(phrase)
                max_wordseq_len = max(max_wordseq_len, len(words))

        logging.info('max_wordseq_len={}'.format(max_wordseq_len))
        computed_params['max_wordseq_len'] = max_wordseq_len

        # Выравниваем все фразы
        pad_func = lpad_wordseq if padding == 'left' else rpad_wordseq
        computed_params['pad_func'] = pad_func
        for sample in samples3:
            sample.anchor_words = pad_func(tokenizer.tokenize(sample.anchor), max_wordseq_len)
            sample.positive_words = pad_func(tokenizer.tokenize(sample.positive), max_wordseq_len)
            sample.negative_words = pad_func(tokenizer.tokenize(sample.negative), max_wordseq_len)
    elif params['repres'] == 'pieces':
        spm_name = 'spm_synonymy({})'.format(params['spm_items'])
        computed_params['spm_name'] = spm_name

        if not os.path.exists(os.path.join(tmp_folder, spm_name + '.model')):
            # Для обучения модели SentencePiece нам нужен текстовый корпус. Изготовим его
            # из имеющихся вариантов предложений в обучающем наборе
            all_texts = set()
            for sample in samples3:
                all_texts.add(sample.anchor)
                all_texts.add(sample.positive)
                all_texts.add(sample.negative)

            sentencepiece_corpus = os.path.join(tmp_folder, 'sentencepiece_corpus.txt')
            with io.open(sentencepiece_corpus, 'w', encoding='utf-8') as wrt:
                for text in all_texts:
                    wrt.write(text)
                    wrt.write(u'\n')

            # Корпус готов, обучаем сегментатор
            logging.info('Train SentencePiece model on {}...'.format(sentencepiece_corpus))
            spm.SentencePieceTrainer.Train(
                '--input={} --model_prefix={} --vocab_size={} --character_coverage=1.0 --model_type=bpe --input_sentence_size=10000000'.format(
                    sentencepiece_corpus, spm_name, params['spm_items']))
            os.rename(spm_name + '.vocab', os.path.join(tmp_folder, spm_name + '.vocab'))
            os.rename(spm_name + '.model', os.path.join(tmp_folder, spm_name + '.model'))

        splitter = spm.SentencePieceProcessor()
        splitter.Load(os.path.join(tmp_folder, spm_name + '.model'))
        computed_params['splitter'] = splitter

        max_wordseq_len = 0
        all_tokens = set([PAD_TOKEN])
        for sample in samples3:
            for phrase in [sample.anchor, sample.positive, sample.negative]:
                tokens = splitter.EncodeAsPieces(phrase)
                max_wordseq_len = max(max_wordseq_len, len(tokens))
                all_tokens.update(tokens)

        logging.info('max_wordseq_len={}'.format(max_wordseq_len))
        computed_params['max_wordseq_len'] = max_wordseq_len

        token2index = {PAD_TOKEN: 0}
        for token in all_tokens:
            if token != PAD_TOKEN:
                token2index[token] = len(token2index)

        computed_params['token2index'] = token2index

        for sample in samples3:
            sample.anchor_words = spm2tokens(splitter, sample.anchor, max_wordseq_len, token2index)
            sample.positive_words = spm2tokens(splitter, sample.positive, max_wordseq_len, token2index)
            sample.negative_words = spm2tokens(splitter, sample.negative, max_wordseq_len, token2index)

    else:
        raise NotImplementedError()

    return samples3, computed_params


def train_model(model, params, train_samples, val_samples, verbose=2):
    batch_size = params['batch_size']
    logging.info('Start training on {} samples with batch_size={}...'.format(len(train_samples), batch_size))
    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                       verbose=verbose,
                                       save_best_only=True,
                                       mode='auto')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=verbose,
                                   mode='auto')

    if params['repres'] == 'words':
        generate_rows = generate_rows_words
    elif params['repres'] == 'pieces':
        generate_rows = generate_rows_pieces

    hist = model.fit_generator(generator=generate_rows(params, computed_params, train_samples, 1),
                               steps_per_epoch=len(train_samples) // batch_size,
                               epochs=NB_EPOCHS,
                               verbose=verbose,
                               callbacks=[model_checkpoint, early_stopping],
                               validation_data=generate_rows(params, computed_params, val_samples, 1),
                               validation_steps=len(val_samples) // batch_size,
                               )

    model.load_weights(weights_path)
    return hist


def eval_model(model, params, computed_params, eval_samples, save_reports):
    val_phrases = set()
    phrase2samples_0 = dict()
    phrase2samples_1 = dict()
    for sample in eval_samples:
        val_phrases.add(sample.anchor)
        val_phrases.add(sample.positive)
        val_phrases.add(sample.negative)

        if sample.anchor in phrase2samples_0:
            phrase2samples_0[sample.anchor].add(sample.negative)
        else:
            phrase2samples_0[sample.anchor] = set([sample.negative])

        if sample.anchor in phrase2samples_1:
            phrase2samples_1[sample.anchor].add(sample.positive)
        else:
            phrase2samples_1[sample.anchor] = set([sample.positive])

    # Генерируем векторы для всех предложений из валидационного набора
    if params['repres'] == 'words':
        tokenizer = computed_params['tokenizer']
        val_phrases = [(s, tokenizer.tokenize(s)) for s in val_phrases]
    elif params['repres'] == 'pieces':
        val_phrases = [(s, None) for s in val_phrases]
    else:
        raise NotImplementedError()

    max_wordseq_len = computed_params['max_wordseq_len']
    batch_size = params['batch_size']

    nb_phrases = len(val_phrases)
    logging.info('Vectorization of %d phrases', nb_phrases)

    if params['repres'] == 'words':
        word_dims = computed_params['word_dims']
        embeddings = computed_params['embeddings']
        X_data = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
        pad_func = computed_params['pad_func']
        for iphrase, phrase in enumerate(val_phrases):
            words = pad_func(phrase[1], max_wordseq_len)
            vectorize_words(words, X_data, iphrase, embeddings)
    elif params['repres'] == 'pieces':
        X_data = np.zeros((nb_phrases, max_wordseq_len, ), dtype=np.int32)
        splitter = computed_params['splitter']
        token2index = computed_params['token2index']
        for iphrase, phrase in enumerate(val_phrases):
            X_data[iphrase, :] = spm2tokens(splitter, phrase[0], max_wordseq_len, token2index)
    else:
        raise NotImplementedError()

    y_pred = model.predict(x=X_data, batch_size=batch_size, verbose=0)

    # Теперь для каждой фразы мы знаем вектор
    phrase2v = dict()
    for i in range(nb_phrases):
        phrase = val_phrases[i][0]
        v = y_pred[i]
        phrase2v[phrase] = v

    group_size = 100
    nb_good = 0
    ranks = []

    for sample in eval_samples:
        samples2 = list()
        samples2.append((sample.anchor, sample.positive, 1))
        samples2.append((sample.anchor, sample.negative, 0))

        # Добавляем негативные сэмплы
        negatives = set()
        for f in val_phrases:
            f = f[0]
            if f != sample.anchor and f != sample.positive and f != sample.negative:
                negatives.add((sample.anchor, f, 0))

        samples2.extend(random.sample(list(negatives), group_size-2))

        # теперь определим похожесть для релевантной пары и для всех нерелевантных в группе
        anchor = phrase2v[sample.anchor]
        sample_rel = []
        for sample2 in samples2:
            # distance between the anchor and the probe
            probe = phrase2v[sample2[1]]
            dist = np.sum(np.square(anchor - probe))
            sample_rel.append((sample2, dist))

        # ранжируем, чтобы самая похожая пара оказалась в начале списка
        sample_rel = sorted(sample_rel, key=lambda z: z[1])

        # Первый сэмпл в этом списке соответствует позитивной паре?
        if sample_rel[0][0][2] == 1:
            nb_good += 1

        # Позиция позитивной пары
        pos = next(i for i, (sample, rel) in enumerate(sample_rel) if sample[2] == 1)
        ranks.append(pos+1)

    # precision@1
    precision1 = nb_good / float(len(eval_samples))

    # mean reciprocal rank
    mrr = np.mean([1.0/rank for rank in ranks])

    return precision1, mrr


def crossval(params, computed_params, samples3):
    kfold_scores = []

    kf = KFold(n_splits=CVFOLDS, random_state=987654321)
    for ifold, (train_index, test_index) in enumerate(kf.split(samples3)):
        logging.info('Starting k-fold #{}'.format(ifold))
        train_samples, val_samples = select_items(samples3, train_index), select_items(samples3, test_index)
        val_samples, eval_samples = train_test_split(val_samples, test_size=0.5, random_state=123456789)
        logging.info('train_samples.count={}'.format(len(train_samples)))
        logging.info('val_samples.count={}'.format(len(val_samples)))
        logging.info('eval_samples.count={}'.format(len(eval_samples)))

        # Для каждого фолда - новая модель
        model, input_anchor, anchor_flow = create_model(params, computed_params)

        # Обучаем
        train_model(model, params, train_samples, val_samples, verbose=0)

        #with open(os.path.join(tmp_folder, 'nn_synonymy_tripleloss.val_loss.csv'), 'w') as wrt:
        #    wrt.write('epoch\tval_loss\n')
        #    for i, valloss in enumerate(hist.history['val_loss']):
        #        wrt.write('{}\t{}\n'.format(i+1, valloss))

        logging.info('Estimating final accuracy on eval dataset...')

        # Создадим сетку, которая будет выдавать вектор одной заданной фразы
        model = Model(inputs=input_anchor, outputs=anchor_flow)
        model.compile(loss='mse', optimizer=params['optimizer'])
        #model.summary()

        precision1, mrr = eval_model(model, params, computed_params, eval_samples, save_reports=True)
        logging.info('precision@1=%g mrr=%g', precision1, mrr)
        kfold_scores.append(precision1)

    score_avg = np.mean(kfold_scores)
    return score_avg

# -------------------------------------------------------------------


NB_EPOCHS = 1000

# Разбор параметров тренировки в командной строке
parser = argparse.ArgumentParser(description='Neural model for short text synonymy')
parser.add_argument('--run_mode', type=str, default='train', choices='train gridsearch query query2'.split(), help='what to do')
parser.add_argument('--input', type=str, default='../data/synonymy_dataset3.csv', help='path to input dataset with triplets')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

args = parser.parse_args()
data_folder = args.data_dir
input_path = args.input
tmp_folder = args.tmp

# настраиваем логирование в файл
ruchatbot.utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_synonymy_tripleloss.log'))

run_mode = args.run_mode
# Варианты значения для параметра run_mode:
# run_mode='train'
# Тренировка модели по заранее приготовленному датасету с парами предложений
# run_mode = 'query'
# В консоли вводятся два предложения, модель оценивает их релевантность.


config_path = os.path.join(tmp_folder, 'nn_synonymy_tripleloss.config')
arch_filepath = os.path.join(tmp_folder, 'nn_synonymy_tripleloss.arch')
weights_path = os.path.join(tmp_folder, 'nn_synonymy_tripleloss.weights')


if run_mode == 'gridsearch':
    logging.info('Start searching the params grid')

    params = dict()

    best_params = None
    best_score = -np.inf

    for repres in ['pieces']:
        params['repres'] = repres

        if params['repres'] == 'pieces':
            for spm_items in [16000]:
                params['spm_items'] = spm_items

                samples3, computed_params = prepare_data(input_path, params, max_samples=100000)

                for token_dim in [64, 128]:
                    params['token_dim'] = token_dim

                    for batch_size in [150]:
                        params['batch_size'] = batch_size

                        for optimizer in ['adam']:
                            params['optimizer'] = optimizer

                            for net_arch in ['lstm']:
                                params['net_arch'] = net_arch

                                for l1 in [0.0]:
                                    params['l1'] = l1

                                    for phrase_dim in [100]:
                                        params['phrase_dim'] = phrase_dim

                                        score = crossval(params, computed_params, samples3)

                                        if score > best_score:
                                            best_score = score
                                            best_params = params.copy()
                                            logging.info('!!! NEW BEST score={} for params={}'.format(best_score,
                                                                                                      get_params_str(
                                                                                                          best_params)))

        elif params['repres'] == 'words':
            params['wordchar2vector_path'] = args.wordchar2vector
            params['word2vector_path'] = os.path.expanduser(args.word2vector)

            samples3, computed_params = prepare_data(input_path, params, max_samples=10000)

            for batch_size in [100]:
                params['batch_size'] = batch_size

                for optimizer in ['adam']:
                    params['optimizer'] = optimizer

                    for net_arch in ['lstm']:
                        params['net_arch'] = net_arch

                        for l1 in [0.0]:
                            params['l1'] = l1

                            for phrase_dim in [100]:
                                params['phrase_dim'] = phrase_dim

                                score = crossval(params, computed_params, samples3)

                                if score > best_score:
                                    best_score = score
                                    best_params = params.copy()
                                    logging.info('!!! NEW BEST score={} for params={}'.format(best_score, get_params_str(best_params)))


    logging.info('Grid search completed, best_score=%f for params=%s', best_score, get_params_str(best_params))


if run_mode == 'train':
    logging.info('Start training')

    params = dict()
    params['batch_size'] = 100
    params['net_arch'] = 'lstm'
    params['repres'] = 'words'
    params['optimizer'] = 'adam'
    params['l1'] = 0.0
    params['phrase_dim'] = 100

    if params['repres'] == 'words':
        params['wordchar2vector_path'] = args.wordchar2vector
        params['word2vector_path'] = os.path.expanduser(args.word2vector)
    else:
        params['token_dim'] = 32
        params['spm_items'] = 8000

    samples3, computed_params = prepare_data(input_path, params, max_samples=1000000)

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    if params['repres'] == 'words':
        model_config = {
            'model': 'nn_synonymy_tripleloss',
            'repres': 'words',
            'max_wordseq_len': computed_params['max_wordseq_len'],
            'w2v_path': params['word2vector_path'],
            'wordchar2vector_path': params['wordchar2vector_path'],
            'PAD_WORD': PAD_WORD,
            'PHRASE_DIM': params['phrase_dim'],
            'padding': padding,
            'arch_filepath': arch_filepath,
            'weights_path': weights_path,
            'word_dims': computed_params['word_dims'],
            'net_arch': params['net_arch'],
        }
    elif params['repres'] == 'pieces':
        model_config = {
            'model': 'nn_synonymy_tripleloss',
            'repres': 'pieces',
            'max_wordseq_len': computed_params['max_wordseq_len'],
            'spm_name': computed_params['spm_name'],
            'token2index': computed_params['token2index'],
            'PAD_WORD': PAD_WORD,
            'PHRASE_DIM': params['phrase_dim'],
            'padding': padding,
            'arch_filepath': arch_filepath,
            'weights_path': weights_path,
            'net_arch': params['net_arch'],
        }
    else:
        raise NotImplementedError()

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    model, input_anchor, anchor_flow = create_model(params, computed_params)

    # разбиваем датасет на обучающую, валидационную и оценочную части.
    # таким образом, финальная оценка будет идти по сэмплам, которые
    # никак не участвовали в обучении модели, в том числе через early stopping
    SEED = 123456
    TEST_SHARE = 0.3
    train_samples, val_samples = train_test_split(samples3, test_size=TEST_SHARE, random_state=SEED)
    val_samples, eval_samples = train_test_split(val_samples, test_size=0.3, random_state=SEED)
    logging.info('train_samples.count={}'.format(len(train_samples)))
    logging.info('val_samples.count={}'.format(len(val_samples)))
    logging.info('eval_samples.count={}'.format(len(eval_samples)))

    hist = train_model(model, params, train_samples, val_samples)

    with open(os.path.join(tmp_folder, 'nn_synonymy_tripleloss.val_loss.csv'), 'w') as wrt:
        wrt.write('epoch\tval_loss\n')
        for i, valloss in enumerate(hist.history['val_loss']):
            wrt.write('{}\t{}\n'.format(i+1, valloss))

    logging.info('Estimating final accuracy on eval dataset...')

    # Создадим сетку, которая будет выдавать вектор одной заданной фразы
    model = Model(inputs=input_anchor, outputs=anchor_flow)
    model.compile(loss='mse', optimizer=params['optimizer'])
    model.summary()

    # Сохраняем архитектуру и веса этой сетки.
    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    model.save_weights(weights_path)
    precision1, mrr = eval_model(model, params, computed_params, eval_samples, save_reports=True)
    logging.info('Evaluation: precision@1=%g   mean reciprocal rank=%g', precision1, mrr)


if run_mode == 'query2':
    # В консоли вводится предложение, для которого в списке smalltalk.txt (или другом)
    # ищутся ближайшие.

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        repres = model_config['repres']
        max_wordseq_len = model_config['max_wordseq_len']
        net_arch = model_config['net_arch']
        padding = model_config['padding']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    if repres == 'words':
        embeddings = WordEmbeddings.load_word_vectors(model_config['wordchar2vector_path'], model_config['w2v_path'])
        word_dims = embeddings.vector_size
        pad_func = lpad_wordseq if padding == 'left' else rpad_wordseq
        tokenizer = Tokenizer()
        tokenizer.load()
    elif repres == 'pieces':
        splitter = spm.SentencePieceProcessor()
        splitter.Load(os.path.join(tmp_folder, model_config['spm_name'] + '.model'))

    # Загрузим эталонные предложения, похожесть на которые будем определять для
    # введенного в консоли предложения.
    phrases2 = set()
    if True:
        for phrase in load_strings_from_yaml(os.path.join(data_folder, 'rules.yaml')):
            phrase2 = u' '.join(tokenizer.tokenize(phrase))
            phrases2.add(phrase2)

    if True:
        with codecs.open(os.path.join(data_folder, 'intents.txt'), 'r', 'utf-8') as rdr:
            for line in rdr:
                phrase = line.strip()
                if len(phrase) > 5 and not phrase.startswith('#') and u'_' not in phrase:
                    phrase2 = u' '.join(tokenizer.tokenize(phrase))
                    phrases2.add(phrase2)

    if True:
        with codecs.open(os.path.join(data_folder, 'faq2.txt'), 'r', 'utf-8') as rdr:
            for line in rdr:
                phrase = line.strip()
                if len(phrase) > 5 and phrase.startswith(u'Q:'):
                    phrase = phrase.replace(u'Q:', u'').strip()
                    phrase2 = u' '.join(tokenizer.tokenize(phrase))
                    phrases2.add(phrase2)

    premises = list(phrases2)
    print('{} phrases in corpus'.format(len(premises)))

    phrases2 = list(phrases2)

    pad_func = lpad_wordseq if padding == 'left' else rpad_wordseq

    while True:
        phrase1 = ruchatbot.utils.console_helpers.input_kbd('phrase:> ').strip().lower()
        if len(phrase1) == 0:
            break

        all_phrases = list(itertools.chain(phrases2, [phrase1]))
        nb_phrases = len(all_phrases)

        if repres == 'words':
            X_data = np.zeros((nb_phrases, max_wordseq_len, word_dims), dtype=np.float32)
            for iphrase, phrase in enumerate(all_phrases):
                words = pad_func(tokenizer.tokenize(phrase), max_wordseq_len)
                vectorize_words(words, X_data, iphrase, embeddings)
        elif repres == 'pieces':
            token2index = model_config['token2index']
            X_data = np.zeros((nb_phrases, max_wordseq_len), dtype=np.int32)
            for iphrase, phrase in enumerate(all_phrases):
                X_data[iphrase, :] = spm2tokens(splitter, phrase, max_wordseq_len, token2index)

        y_pred = model.predict(x=X_data, verbose=0)

        # Теперь для каждой фразы мы знаем вектор
        phrase2v = dict()
        for i in range(nb_phrases):
            phrase = all_phrases[i]
            v = y_pred[i]
            phrase2v[phrase] = v

        # Можем оценить близость введенной фразы к каждому образцу в smalltalk.txt
        phrase_sims = []
        v1 = phrase2v[phrase1]
        for phrase2 in phrases2:
            v2 = phrase2v[phrase2]
            sim = v_cosine(v1, v2)
            phrase_sims.append((phrase2, sim))

        # Выведем top 10 ближайших фраз
        phrase_sims = sorted(phrase_sims, key=lambda z: -z[1])
        for phrase, sim in phrase_sims[:10]:
            print(u'{:6.4f} {}'.format(sim, phrase))

if run_mode == 'query':
    # В консоли вводятся два предложения, модель выдает их похожесть

    # Грузим конфигурацию модели, веса и т.д.
    with open(config_path, 'r') as f:
        model_config = json.load(f)
        repres = model_config['repres']
        max_wordseq_len = model_config['max_wordseq_len']
        net_arch = model_config['net_arch']
        padding = model_config['padding']

    with open(arch_filepath, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_path)

    if repres == 'words':
        embeddings = WordEmbeddings.load_word_vectors(model_config['wordchar2vector_path'], model_config['w2v_path'])
        word_dims = embeddings.vector_size
        pad_func = lpad_wordseq if padding == 'left' else rpad_wordseq
        tokenizer = Tokenizer()
        tokenizer.load()
    elif repres == 'pieces':
        splitter = spm.SentencePieceProcessor()
        splitter.Load(os.path.join(tmp_folder, model_config['spm_name'] + '.model'))

    while True:
        phrase1 = ruchatbot.utils.console_helpers.input_kbd('phrase #1:> ').strip().lower()
        if len(phrase1) == 0:
            break

        phrase2 = ruchatbot.utils.console_helpers.input_kbd('phrase #2:> ').strip().lower()
        if len(phrase2) == 0:
            break

        all_phrases = list(itertools.chain([phrase1], [phrase2]))
        nb_phrases = len(all_phrases)

        if repres == 'words':
            X_data = np.zeros((2, max_wordseq_len, word_dims), dtype=np.float32)
            for iphrase, phrase in enumerate(all_phrases):
                words = pad_func(tokenizer.tokenize(phrase), max_wordseq_len)
                vectorize_words(words, X_data, iphrase, embeddings)
        elif repres == 'pieces':
            token2index = model_config['token2index']
            X_data = np.zeros((2, max_wordseq_len), dtype=np.int32)
            for iphrase, phrase in enumerate(all_phrases):
                X_data[iphrase, :] = spm2tokens(splitter, phrase, max_wordseq_len, token2index)

        y_pred = model.predict(x=X_data, verbose=0)

        v1 = y_pred[0]
        v2 = y_pred[1]
        sim = v_cosine(v1, v2)
        print(u'{:6.4f}'.format(sim))

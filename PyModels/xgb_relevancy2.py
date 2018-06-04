# -*- coding: utf-8 -*-
"""
Тренировка модели определения релевантности предпосылки и вопроса.
Модель используется в проекте чат-бота https://github.com/Koziev/chatbot
Используется XGBoost.

ЭКСПЕРИМЕНТ: добавляем косинусные меры похожести по среднему вектору и метрику wmd,
и другие фичи (см. enabled_features)
"""

from __future__ import division
from __future__ import print_function

import gc
import itertools
import json
import os
import sys
import argparse
import codecs

import gensim
import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.merge import concatenate, add, multiply
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization

from utils.segmenter import Segmenter
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup
from utils.phrase_splitter import PhraseTokenizer


config_filename = 'xgb_relevancy2.config'

parser = argparse.ArgumentParser(description='XGB classifier for text relevance estimation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train evaluate query query2')
parser.add_argument('--shingle_len', type=str, default='3', help='shingle length')
parser.add_argument('--max_depth', type=int, default=6, help='max depth parameter for XGBoost')
parser.add_argument('--eta', type=float, default=0.20, help='eta (learning rate) parameter for XGBoost')
parser.add_argument('--input', type=str, default='../data/premise_question_relevancy.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')
parser.add_argument('--word2vector', type=str, default='/home/eek/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=32.model', help='path to word2vector model file')
parser.add_argument('--subsample', type=float, default=1.00, help='"subsample" parameter for XGBoost')
parser.add_argument('--lemmatize', type=int, default=0, help='lemmatize phrases before extracting the shingles')
parser.add_argument('--use_sent2vec', type=int, default=0, help='Use sent2vec sentence representation as features')
parser.add_argument('--use_skip_shingles', type=int, default=0, help='Add skip shingles to feature set')
parser.add_argument('--min_skipshingle_dist', type=int, default=1)
parser.add_argument('--max_skipshingle_dist', type=int, default=5)
parser.add_argument('--use_shingles', type=int, default=1)

args = parser.parse_args()

input_path = args.input
tmp_folder = args.tmp
data_folder = args.data_dir
word2vector_path = args.word2vector
run_mode = args.run_mode
lemmatize = args.lemmatize
subsample = args.subsample

# основной настроечный параметр модели - используемые длины символьных N-грамм (шинглов)
shingle_lens = [int(s) for s in args.shingle_len.split(',')]

max_depth = args.max_depth
eta = args.eta
use_sent2vec = args.use_sent2vec
use_skip_shingles = args.use_skip_shingles
use_shingles = args.use_shingles
min_skipshingle_dist = args.min_skipshingle_dist
max_skipshingle_dist = args.max_skipshingle_dist

# -------------------------------------------------------------------

BEG_WORD = '\b'
END_WORD = '\n'


def ngrams(s, shingle_len):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(shingle_len)])]


def ngrams2(s, shingle_lens):
    return list(itertools.chain(*(ngrams(s, shingle_len) for shingle_len in shingle_lens)))


def words2str(words):
    return u' '.join(itertools.chain([BEG_WORD], words, [END_WORD]))


def get_average_vector(words, w2v):
    v = None
    denom = 0
    for iword, word in enumerate(words):
        if word in w2v:
            denom += 1
            if v is None:
                v = np.array(w2v[word])
            else:
                v += w2v[word]

    return v/denom if denom>0 else None


def v_cosine(a, b):
    if a is None or b is None:
        return 0.0

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


def vectorize_words(words, X_data, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            X_data[irow, iword, :] = word2vec[word]


class EnabledFeatures(object):
    def __init__(self):
        self.use_shingles = True
        self.average_w2v = False
        self.average_w2v_cosine = False
        self.wmd = False
        self.sent2vec = False
        self.use_skip_shingles = True
        self.min_skipshingle_dist = 1
        self.max_skipshingle_dist = 5

    def calc_nb_features(self, nb_shingles, w2v_dim, sent2vec_dim):
        nb_features = 0

        if self.use_shingles:
           nb_features += nb_shingles * 3

        if self.average_w2v_cosine:
            nb_features += 1

        if self.wmd:
            nb_features += 1

        if self.average_w2v:
            nb_features += w2v_dim*2

        if self.sent2vec:
            nb_features += sent2vec_dim*3

        return nb_features

    def from_dict(self, d):
        self.average_w2v_cosine = bool(d['average_w2v_cosine'])
        self.wmd = bool(d['wmd'])
        self.average_w2v = bool(d['average_w2v'])
        self.sent2vec = bool(d['sent2vec']) if 'sent2vec' in d else False
        self.use_shingles = bool(d['use_shingles'])
        self.use_skip_shingles = bool(d['use_skip_shingles'])
        self.min_skipshingle_dist = int(d['min_skipshingle_dist'])
        self.max_skipshingle_dist = int(d['max_skipshingle_dist'])

    def to_dict(self):
        return {'average_w2v_cosine': self.average_w2v_cosine,
                'wmd': self.wmd,
                'average_w2v': self.average_w2v,
                'sent2vec': self.sent2vec,
                'use_shingles': self.use_shingles,
                'use_skip_shingles': self.use_skip_shingles,
                'min_skipshingle_dist': self.min_skipshingle_dist,
                'max_skipshingle_dist': self.max_skipshingle_dist
                }

    def X_field_type(self):
        if not self.use_shingles and self.sent2vec:
            return 'float32'
        if self.average_w2v or self.average_w2v_cosine or self.wmd:
            return 'float32'
        else:
            return 'bool'

    def needs_w2v(self):
        return self.average_w2v or self.average_w2v_cosine or self.wmd


def split_to_shingles_with_pos(phrase_str, shingle_len):
    shingle_pos_list = []
    for i in range(0, len(phrase_str)-shingle_len):
        shingle = phrase_str[i:i+shingle_len]
        shingle_pos_list.append((i, shingle))

    return shingle_pos_list


def extract_shingles(phrase_str, shingle_lens, enabled_features):
    all_shingles = set()
    for shingle_len in shingle_lens:
        all_shingles.update(ngrams(phrase_str, shingle_len))

    if enabled_features.use_skip_shingles:
        shingle_pos_list = split_to_shingles_with_pos(phrase_str, 3)

        for pos1, shingle1 in shingle_pos_list:
            for pos2, shingle2 in shingle_pos_list:
                if pos2 > pos1:
                    if enabled_features.max_skipshingle_dist >= (pos2-pos1) >= enabled_features.min_skipshingle_dist:
                        shingle = shingle1+u'(..)'.format(pos2-pos1)+shingle2
                        all_shingles.add(shingle)

    return all_shingles


def vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, shingle2id,
                       premise_words, question_words,
                       premise_sent2vec, question_sent2vec,
                       w2v,
                       enabled_features):
    icol = 0

    if enabled_features.use_shingles:
        ps = set(premise_shingles)
        qs = set(question_shingles)
        common_shingles = ps & qs
        notmatched_ps = ps - qs
        notmatched_qs = qs - ps

        nb_shingles = len(shingle2id)

        for shingle in common_shingles:
            if shingle in shingle2id:
                X_data[idata, icol+shingle2id[shingle]] = True
        icol += nb_shingles

        for shingle in notmatched_ps:
            if shingle in shingle2id:
                X_data[idata, icol+shingle2id[shingle]] = True
        icol += nb_shingles

        for shingle in notmatched_qs:
            if shingle in shingle2id:
                X_data[idata, icol+shingle2id[shingle]] = True
        icol += nb_shingles

    if enabled_features.average_w2v or enabled_features.average_w2v_cosine:
        v1 = get_average_vector(premise_words, w2v)
        v2 = get_average_vector(question_words, w2v)
        w2v_dim = len(v1)

        if enabled_features.average_w2v_cosine:
            k1 = v_cosine(v1, v2)
            X_data[idata, icol] = k1
            icol += 1

        if enabled_features.average_w2v:
            X_data[idata, icol:icol+w2v_dim] = v1
            icol += w2v_dim

            X_data[idata, icol:icol + w2v_dim] = v2
            icol += w2v_dim

    if enabled_features.wmd:
        k2 = w2v.wmdistance(premise_words, question_words)
        X_data[idata, icol] = k2
        icol += 1

    if enabled_features.sent2vec:
        sent2vec_dim = len(premise_sent2vec)

        for j in range(sent2vec_dim):
            if premise_sent2vec[j] > 0:
                X_data[idata, icol+j] = premise_sent2vec[j]
        icol += sent2vec_dim

        for j in range(sent2vec_dim):
            if question_sent2vec[j] > 0:
                X_data[idata, icol+j] = question_sent2vec[j]
        icol += sent2vec_dim

        for j in range(sent2vec_dim):
            if question_sent2vec[j] != premise_sent2vec[j]:
                X_data[idata, icol+j] = question_sent2vec[j] - premise_sent2vec[j]
        icol += sent2vec_dim


def load_models(enabled_features, word2vector_path):
    word2vec = None
    sent2vec = None
    w2v = None
    sent2vec_dim = None
    threshold = None
    max_wordseq_len = None
    word_dims = None

    if enabled_features.sent2vec:
        model_folder = tmp_folder

        # некоторые необходимые для векторизации предложений параметры
        # сохранены программой nn_relevancy.py в файле sent2vec.config
        with open(os.path.join(model_folder, 'sent2vec.config'), 'r') as f:
            model_config = json.load(f)
            max_wordseq_len = model_config['max_wordseq_len']
            word2vector_path = model_config['w2v_path']
            wordchar2vector_path = model_config['wordchar2vector_path']
            word_dims = int(model_config['word_dims'])
            sent2vec_dim = int(model_config['sent2vec_dim'])
            threshold = float(model_config['threshold'])
            arch_filepath2 = model_config['arch_filepath']
            weights_path2 = model_config['weights_path']

        # Загружаем модель sent2vec
        with open(arch_filepath2, 'r') as f:
            sent2vec = model_from_json(f.read())

        sent2vec.load_weights(weights_path2)

        print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
        wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
        wc2v_dims = len(wc2v.syn0[0])
        print('wc2v_dims={0}'.format(wc2v_dims))

        print('Loading the w2v model {}'.format(word2vector_path))
        w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
        w2v_dims = len(w2v.syn0[0])
        print('w2v_dims={0}'.format(w2v_dims))

        word2vec = dict()
        for word in wc2v.vocab:
            v = np.zeros(word_dims)
            v[w2v_dims:] = wc2v[word]
            if word in w2v:
                v[:w2v_dims] = w2v[word]

            word2vec[word] = v

        del wc2v
        gc.collect()
    elif enabled_features.needs_w2v():
        print('Loading the w2v model {}'.format(word2vector_path))
        w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=True)
        w2v_dims = len(w2v.syn0[0])
        print('w2v_dims={0}'.format(w2v_dims))

    return w2v, word2vec, sent2vec, word_dims, max_wordseq_len, sent2vec_dim, threshold

# -------------------------------------------------------------------

if run_mode == 'train':
    model_folder = tmp_folder
    enabled_features = EnabledFeatures()
    enabled_features.sent2vec = use_sent2vec
    enabled_features.use_skip_shingles = use_skip_shingles
    enabled_features.min_skipshingle_dist = min_skipshingle_dist
    enabled_features.max_skipshingle_dist = max_skipshingle_dist

    enabled_features.average_w2v_cosine = False
    enabled_features.wmd = False
    enabled_features.use_shingles = use_shingles

    tokenizer = PhraseTokenizer.create_splitter(lemmatize)

    w2v, word2vec, sent2vec, word_dims, max_wordseq_len, sent2vec_dim, threshold = load_models(enabled_features, word2vector_path)

    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    print('samples.count={}'.format(df.shape[0]))

    phrases = []
    ys = []
    weights = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extraction of phrases'):
        weights.append(row['weight'])
        phrase1 = row['premise']
        phrase2 = row['question']
        words1 = words2str( tokenizer.tokenize(phrase1) )
        words2 = words2str( tokenizer.tokenize(phrase2) )

        y = row['relevance']
        if y in (0, 1):
            ys.append(y)
            phrases.append( (words1, words2, phrase1, phrase2) )

    nb_patterns = len(ys)
    print('nb_patterns={}'.format(nb_patterns))

    # Вопросы и предпосылки прогоним через sent2vec модель
    premises_sent2vec = []
    questions_sent2vec = []
    sent2vec_dims = 0
    if enabled_features.sent2vec:
        X_premises = np.zeros((nb_patterns, max_wordseq_len, word_dims), dtype=np.float32)
        X_questions = np.zeros((nb_patterns, max_wordseq_len, word_dims), dtype=np.float32)

        for isent, phrase12 in tqdm.tqdm(enumerate(phrases), total=nb_patterns, desc='sent2vec'):
            premise = phrase12[0]
            question = phrase12[1]

            premise_words = tokenizer.tokenize(premise)
            if len(premise_words) > max_wordseq_len:
                #print(u'Premise {} contains more than max_wordseq_len={} words'.format(premise, max_wordseq_len))
                premise_words = premise_words[:max_wordseq_len]
            vectorize_words(premise_words, X_premises, isent, word2vec)

            question_words = tokenizer.tokenize(question)
            if len(question_words) > max_wordseq_len:
                #print(u'Question {} contains more than max_wordseq_len={} words'.format(question, max_wordseq_len))
                question_words = question_words[:max_wordseq_len]
            vectorize_words(question_words, X_questions, isent, word2vec)

        nn_batch_size = 1000
        print('Running sent2vector model for {} premises...'.format(nb_patterns))
        premises_sent2vec = sent2vec.predict(X_premises, batch_size=nn_batch_size, verbose=1)

        print('Running sent2vector model for {} questions...'.format(nb_patterns))
        questions_sent2vec = sent2vec.predict(X_questions, batch_size=nn_batch_size, verbose=1)

        # Выполняем бинаризацию векторов предложений.

        if False:
            # начало отладки
            print('premises_sent2vec[0]=', premises_sent2vec[0,:])
            threshold1 = threshold
            for z in range(10):
                threshold1 = threshold1*0.1
                premises_sent2vec1 = (premises_sent2vec > threshold1).astype(np.int_)
                n1 = np.sum(premises_sent2vec1)
                sparsity = float(n1)/(np.prod(premises_sent2vec1.shape))
                print('threshold1={:<8.4e} premises_sent2vec sparsity={}'.format(threshold1, sparsity))
            #exit(0)
            # конец отладки

            premises_sent2vec = (premises_sent2vec > threshold).astype(np.int_)
            n1 = np.sum(premises_sent2vec)
            sparsity = float(n1)/(np.prod(premises_sent2vec.shape))
            print('premises_sent2vec sparsity={}'.format(sparsity))

            questions_sent2vec = (questions_sent2vec > threshold).astype(np.int_)
            n1 = np.sum(questions_sent2vec)
            sparsity = float(n1)/(np.prod(questions_sent2vec.shape))
            print('questions_sent2vec sparsity={}'.format(sparsity))

    nb_shingles = 0
    shingle2id = None
    if enabled_features.use_shingles:
        all_shingles = set()
        for i,record in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Shingles'):
            for phrase in [record['premise'], record['question']]:
                words = tokenizer.tokenize(phrase)
                wx = words2str(words)
                all_shingles.update(extract_shingles(wx, shingle_lens, enabled_features))

        nb_shingles = len(all_shingles)
        print('nb_shingles={}'.format(nb_shingles))
        shingle2id = dict([(s,i) for i,s in enumerate(all_shingles)])

    nb_features = enabled_features.calc_nb_features(nb_shingles, word_dims, sent2vec_dim)
    print('nb_features={}'.format(nb_features))

    X_data = lil_matrix((nb_patterns, nb_features), dtype=enabled_features.X_field_type())
    y_data = []

    for idata, (phrase12, y12) in tqdm.tqdm(enumerate(itertools.izip(phrases, ys)), total=nb_patterns, desc='Vectorization'):
        premise = phrase12[0]
        question = phrase12[1]
        y = y12

        y_data.append(y)

        if enabled_features.use_shingles:
            premise_shingles = extract_shingles(premise, shingle_lens, enabled_features)  # ngrams2(premise, shingle_lens)
            question_shingles = extract_shingles(question, shingle_lens, enabled_features)  # ngrams2(question, shingle_lens)
        else:
            premise_shingles = None
            question_shingles = None

        if enabled_features.sent2vec:
            premise_sent2vec = premises_sent2vec[idata]
            question_sent2vec = questions_sent2vec[idata]
        else:
            premise_sent2vec = None
            question_sent2vec = None

        vectorize_sample_x( X_data, idata,
                            premise_shingles, question_shingles, shingle2id,
                            premise, question,
                            premise_sent2vec, question_sent2vec,
                            w2v, enabled_features)

    nb_0 = len(filter(lambda y: y == 0, y_data))
    nb_1 = len(filter(lambda y: y == 1, y_data))

    print('nb_0={}'.format(nb_0))
    print('nb_1={}'.format(nb_1))

    # ------------------------------------------------------------------------

    SEED = 123456
    TEST_SHARE = 0.2
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_data, y_data, weights, test_size=TEST_SHARE, random_state=SEED)

    D_train = xgboost.DMatrix(data=X_train, label=y_train, weight=w_train, silent=0)
    D_val = xgboost.DMatrix(data=X_val, label=y_val, weight=w_val, silent=0)

    del X_train
    del X_val
    del X_data
    del df
    gc.collect()

    xgb_params = {
        'booster': 'gbtree',
        # 'n_estimators': _n_estimators,
        'subsample': subsample,
        'max_depth': max_depth,
        'seed': 123456,
        'min_child_weight': 1,
        'eta': eta,
        'gamma': 0.01,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'eval_metric': 'error',
        'objective': 'binary:logistic',
        'silent': 1,
        # 'updater': 'grow_gpu'
    }

    print('Train model...')
    cl = xgboost.train(xgb_params,
                       D_train,
                       evals=[(D_val, 'val')],
                       num_boost_round=10000,
                       verbose_eval=50,
                       early_stopping_rounds=50)

    print('Training is finished')
    y_pred = cl.predict(D_val)
    y_pred = (y_pred >= 0.5).astype(np.int)
    acc = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred)
    print('val acc={}'.format(acc))

    # из-за сильного дисбаланса (в пользу исходов с y=0) оценивать качество
    # получающейся модели лучше по f1
    f1 = sklearn.metrics.f1_score(y_true=y_val, y_pred=y_pred)
    print('val f1={}'.format(f1))


    model_filename = os.path.join( tmp_folder, 'xgb_relevancy2.model' )

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'model': 'xgb',
                    'shingle2id': shingle2id,
                    'model_filename': model_filename,
                    'shingle_lens': shingle_lens,
                    'nb_features': nb_features,
                    'word2vector_path': word2vector_path,
                    'enabled_features': enabled_features.to_dict(),
                    'lemmatize': lemmatize
                   }

    with open(os.path.join(tmp_folder, config_filename), 'w') as f:
        json.dump(model_config, f)

    cl.save_model( model_filename )


if run_mode == 'query':
    # Проверка модели по парам фраз, вводимым в консоли.

    # Загружаем данные обученной модели.
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    xgb_enabled_features = EnabledFeatures()
    xgb_enabled_features.from_dict(model_config['enabled_features'])

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_lens = model_config['shingle_lens']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmatize = model_config['lemmatize']

    tokenizer = PhraseTokenizer.create_splitter(xgb_relevancy_lemmatize)

    xgb_relevancy = xgboost.Booster()
    xgb_relevancy.load_model(model_config['model_filename'])

    while True:
        premise = raw_input('premise:> ').decode(sys.stdout.encoding).strip().lower()
        if len(premise) == 0:
            break

        question = raw_input('question:> ').decode(sys.stdout.encoding).strip().lower()
        if len(question) == 0:
            break

        premise_words = tokenizer.tokenize(premise)
        question_words = tokenizer.tokenize(question)

        X_data = lil_matrix((1, xgb_relevancy_nb_features), dtype='bool')

        premise_wx = words2str(premise_words)
        question_wx = words2str(question_words)

        if xgb_enabled_features.use_shingles:
            premise_shingles = extract_shingles(premise_wx, xgb_relevancy_shingle_lens, xgb_enabled_features)
            question_shingles = extract_shingles(question_wx, xgb_relevancy_shingle_lens, xgb_enabled_features)
        else:
            premise_shingles = None
            question_shingles = None

        if enabled_features.sent2vec:
            premise_sent2vec = premises_sent2vec[idata]
            question_sent2vec = questions_sent2vec[idata]
        else:
            premise_sent2vec = None
            question_sent2vec = None

        vectorize_sample_x(X_data, 0, premise_shingles, question_shingles, xgb_relevancy_shingle2id,
                           premise_words, question_words,
                           premise_sent2vec, question_sent2vec,
                           w2v, xgb_enabled_features)

        D_data = xgboost.DMatrix(X_data)
        y_probe = xgb_relevancy.predict(D_data)
        print(y_probe[0])


if run_mode == 'query2':
    # С клавиатуры задается путь к файлу с предложениями, и второе предложение.
    # Модель делает оценку релевантности для каждого предложения в файле и введенного предложения,
    # и сохраняет список оценок с сортировкой.

    # загружаем данные обученной модели
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    xgb_enabled_features = EnabledFeatures()
    xgb_enabled_features.from_dict(model_config['enabled_features'])

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_lens = model_config['shingle_lens']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmatize = model_config['lemmatize']

    tokenizer = PhraseTokenizer.create_splitter(xgb_relevancy_lemmatize)

    xgb_relevancy = xgboost.Booster()
    xgb_relevancy.load_model(model_config['model_filename'])

    path1 = raw_input('path to text file with phrases:\n> ').decode(sys.stdout.encoding).strip().lower()

    phrases1 = []
    segm_mode = raw_input('Use EOL markers (1) or segmenter (2) to split file to sentences?').strip()

    max_nb_facts = int(raw_input('maximum number of samples to read from file (-1 means all):\n> ').strip())
    if max_nb_facts == -1:
        max_nb_facts = 10000000

    if segm_mode == 2:
        segmenter = Segmenter()
        phrases0 = segmenter.split(codecs.open(path1, 'r', 'utf-8').readlines())
        for phrase in enumerate(phrases):
            words = tokenizer.tokenize(phrase)
            if len(words) > 0:
                phrases1.append(words)
            if len(phrases1) >= max_nb_facts:
                break
    else:
        with codecs.open(path1, 'r', 'utf-8') as rdr:
            for phrase in rdr:
                words = tokenizer.tokenize(phrase)
                if len(words) > 0:
                    phrases1.append(words)
                if len(phrases1) >= max_nb_facts:
                    break

    nb_phrases = len(phrases1)
    print(u'{1} phrases are loaded from {0}'.format(path1, nb_phrases))

    while True:
        # нужна чистая матрица
        X_data = lil_matrix((nb_phrases, xgb_relevancy_nb_features), dtype='bool')

        # вводим проверяемую фразу (вопрос) с консоли
        phrase2 = raw_input('phrase #2:> ').decode(sys.stdout.encoding).strip().lower()
        if len(phrase2) == 0:
            break

        question_words = tokenizer.tokenize(phrase2)

        for iphrase, premise_words in enumerate(phrases1):
            premise_wx = words2str(premise_words)
            question_wx = words2str(question_words)

            if xgb_enabled_features.use_shingles:
                premise_shingles = extract_shingles(premise_wx, xgb_relevancy_shingle_lens, xgb_enabled_features)
                question_shingles = extract_shingles(question_wx, xgb_relevancy_shingle_lens, xgb_enabled_features)
            else:
                premise_shingles = None
                question_shingles = None

            if enabled_features.sent2vec:
                premise_sent2vec = premises_sent2vec[idata]
                question_sent2vec = questions_sent2vec[idata]
            else:
                premise_sent2vec = None
                question_sent2vec = None

            vectorize_sample_x(X_data, iphrase, premise_shingles, question_shingles, xgb_relevancy_shingle2id,
                               premise_words, question_words,
                               premise_sent2vec, question_sent2vec,
                               w2v, xgb_enabled_features)

        D_data = xgboost.DMatrix(X_data)
        y_probe = xgb_relevancy.predict(D_data)

        phrase_indeces = sorted( range(nb_phrases), key=lambda i:-y_probe[i] )

        print('Phrases ranked by descending relevance:')
        for phrase_index in phrase_indeces[:10]:
            print(u'{:4f}\t{}'.format(y_probe[phrase_index], u' '.join(phrases1[phrase_index])))


if run_mode == 'evaluate':
    # Загружаем данные обученной модели.
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    xgb_enabled_features = EnabledFeatures()
    xgb_enabled_features.from_dict(model_config['enabled_features'])

    word2vector_path = model_config['word2vector_path']

    w2v, word2vec, sent2vec, word_dims, max_wordseq_len, sent2vec_dim, threshold = load_models(xgb_enabled_features, word2vector_path)

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_lens = model_config['shingle_lens']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmatize = model_config['lemmatize']

    tokenizer = PhraseTokenizer.create_splitter(xgb_relevancy_lemmatize)

    xgb_relevancy = xgboost.Booster()
    xgb_relevancy.load_model(model_config['model_filename'])

    # Оценка качества натренированной модели на специальном наборе вопросов и ожидаемых выборов предпосылок
    # из тренировочного набора.
    eval_data = EvaluationDataset(0, tokenizer)
    eval_data.load(data_folder)

    nb_good = 0
    nb_bad = 0

    with codecs.open(os.path.join(tmp_folder, 'xgb_relevancy2.evaluation.txt'), 'w', 'utf-8') as wrt:
        for irecord, phrases in eval_data.generate_groups():
            # будем оценивать качество выбора правильной предпосылки для заданного вопроса.
            # у нас есть множество альтернативных предпосылок, среди которых одна - релевантная.
            # вопрос - одинаковый для всех предпосылок.
            nb_samples = len(phrases)

            # тензор для XGB классификатора
            X_data = lil_matrix((nb_samples, xgb_relevancy_nb_features), dtype='bool')

            # тензор для векторизации предложений с помощью sent2vec модели
            if xgb_enabled_features.sent2vec:
                X_sent2vec = np.zeros((1, max_wordseq_len, word_dims), dtype=np.float32)

            for irow, (premise_words, question_words) in enumerate(phrases):
                premise_wx = words2str(premise_words)
                question_wx = words2str(question_words)

                if xgb_enabled_features.use_shingles:
                    premise_shingles = extract_shingles(premise_wx, xgb_relevancy_shingle_lens, xgb_enabled_features)
                    question_shingles = extract_shingles(question_wx, xgb_relevancy_shingle_lens, xgb_enabled_features)
                else:
                    premise_shingles = None
                    question_shingles = None

                if xgb_enabled_features.sent2vec:
                    X_sent2vec.fill(0)
                    vectorize_words(premise_words, X_sent2vec, 0, word2vec)
                    premise_sent2vec = sent2vec.predict(X_sent2vec, verbose=0)[0]

                    X_sent2vec.fill(0)
                    vectorize_words(question_words, X_sent2vec, 0, word2vec)
                    question_sent2vec = sent2vec.predict(X_sent2vec, verbose=0)[0]

                    # бинаризация векторов предложений с учетом threshold'а из модели sent2vec
                    premise_sent2vec = (premise_sent2vec > threshold).astype(np.int_)
                    question_sent2vec = (question_sent2vec > threshold).astype(np.int_)
                else:
                    premise_sent2vec = None
                    question_sent2vec = None

                vectorize_sample_x(X_data, irow, premise_shingles, question_shingles,
                                   xgb_relevancy_shingle2id,
                                   premise_words, question_words,
                                   premise_sent2vec, question_sent2vec,
                                   w2v, xgb_enabled_features)

            D_data = xgboost.DMatrix(X_data)
            y_pred = xgb_relevancy.predict(D_data)

            # предпосылка с максимальной релевантностью
            max_index = np.argmax(y_pred)
            selected_premise = u' '.join(phrases[max_index][0]).strip()

            # эта выбранная предпосылка соответствует одному из вариантов
            # релевантных предпосылок в этой группе?
            if eval_data.is_relevant_premise(irecord, selected_premise):
                nb_good += 1
                print(EvaluationMarkup.ok_color + EvaluationMarkup.ok_bullet + EvaluationMarkup.close_color, end='')
                wrt.write(EvaluationMarkup.ok_bullet)
            else:
                nb_bad += 1
                print(EvaluationMarkup.fail_color + EvaluationMarkup.fail_bullet + EvaluationMarkup.close_color, end='')
                wrt.write(EvaluationMarkup.fail_bullet)

            max_sim = np.max(y_pred)

            question_words = phrases[0][1]
            message_line = u'{:<40} {:<40} {}/{}'.format(u' '.join(question_words), u' '.join(phrases[max_index][0]), y_pred[max_index], y_pred[0])
            print(message_line)
            wrt.write(message_line+u'\n')

            # для отладки: top релевантных вопросов
            if False:
                print(u'Most similar premises for question {}'.format(u' '.join(question)))
                yy = [(y_pred[i], i) for i in range(len(y_pred))]
                yy = sorted(yy, key=lambda z:-z[0])

                for sim, index in yy[:5]:
                    print(u'{:.4f} {}'.format(sim, u' '.join(phrases[index][0])))

    # Итоговая точность выбора предпосылки.
    accuracy = float(nb_good)/float(nb_good+nb_bad)
    print('accuracy={}'.format(accuracy))

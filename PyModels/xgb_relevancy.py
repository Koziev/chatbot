# -*- coding: utf-8 -*-
"""
Тренировка модели определения релевантности предпосылки и вопроса.
Модель используется в проекте чат-бота https://github.com/Koziev/chatbot
Используется XGBoost.
"""

from __future__ import division
from __future__ import print_function

import gc
import itertools
import json
import os
import sys
import argparse

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer





parser = argparse.ArgumentParser(description='Neural model for text relevance estimation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train evaluate query query2')
parser.add_argument('--shingle_len', type=int, default=3, help='shingle length')
parser.add_argument('--max_depth', type=int, default=4, help='max depth parameter for XGBoost')
parser.add_argument('--eta', type=float, default=0.20, help='eta (learning rate) parameter for XGBoost')
parser.add_argument('--input', type=str, default='../data/premise_question_relevancy.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')

args = parser.parse_args()

input_path = args.input
tmp_folder = args.tmp
run_mode = args.run_mode

# основной настроечный параметр модели - длина символьных N-грамм (шинглов)
shingle_len = args.shingle_len

max_depth = args.max_depth
eta = args.eta

# -------------------------------------------------------------------

BEG_WORD = '\b'
END_WORD = '\n'


def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    return u' '.join(itertools.chain([BEG_WORD], words, [END_WORD]))


def vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, shingle2id):
    ps = set(premise_shingles)
    qs = set(question_shingles)
    common_shingles = ps & qs
    notmatched_ps = ps - qs
    notmatched_qs = qs - ps

    nb_shingles = len(shingle2id)

    icol = 0
    for shingle in common_shingles:
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_ps:
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_qs:
        X_data[idata, icol+shingle2id[shingle]] = True

# -------------------------------------------------------------------

tokenizer = Tokenizer()


if run_mode == 'train':
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    print('samples.count={}'.format(df.shape[0]))

    all_shingles = set()

    for i,record in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Shingles'):
        for phrase in [record['premise'], record['question']]:
            words = tokenizer.tokenize(phrase)
            wx = words2str(words)
            all_shingles.update(ngrams(wx, shingle_len))

    nb_shingles = len(all_shingles)
    print('nb_shingles={}'.format(nb_shingles))

    shingle2id = dict([(s,i) for i,s in enumerate(all_shingles)])

    phrases = []
    ys = []
    weights = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
        weights.append(row['weight'])
        phrase1 = row['premise']
        phrase2 = row['question']
        words1 = words2str( tokenizer.tokenize(phrase1) )
        words2 = words2str( tokenizer.tokenize(phrase2) )

        y = row['relevance']
        if y in (0,1):
            ys.append(y)
            phrases.append( (words1, words2, phrase1, phrase2) )

    nb_patterns = len(ys)

    nb_features = nb_shingles*3
    X_data = lil_matrix((nb_patterns, nb_features), dtype='bool')
    y_data = []

    for idata, (phrase12, y12) in tqdm.tqdm(enumerate(itertools.izip(phrases, ys)), total=nb_patterns, desc='Vectorization'):
        premise = phrase12[0]
        question = phrase12[1]
        y = y12

        y_data.append(y)

        premise_shingles = ngrams(premise, shingle_len)
        question_shingles = ngrams(question, shingle_len)
        vectorize_sample_x( X_data, idata, premise_shingles, question_shingles, shingle2id )

    nb_0 = len(filter(lambda y:y==0, y_data))
    nb_1 = len(filter(lambda y:y==1, y_data))

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
        'subsample': 1.0,
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
                       num_boost_round=1000,
                       verbose_eval=50,
                       early_stopping_rounds=50)

    print('Training is finished')
    y_pred = cl.predict(D_val)
    y_pred = (y_pred >= 0.5).astype(np.int)
    score = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred)
    print('score={}'.format(score))


    model_filename = os.path.join( tmp_folder, 'xgb_relevancy.model' )

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'model': 'xgb',
                    'shingle2id': shingle2id,
                    'model_filename': model_filename,
                    'shingle_len': shingle_len,
                    'nb_features': nb_features,
                   }

    with open(os.path.join(tmp_folder,'xgb_relevancy.config'), 'w') as f:
        json.dump(model_config, f)

    cl.save_model( model_filename )

if run_mode == 'query':
    with open(os.path.join(tmp_folder, 'xgb_relevancy.config'), 'r') as f:
        model_config = json.load(f)

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']

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

        premise_shingles = set(ngrams(premise_wx, xgb_relevancy_shingle_len))
        question_shingles = set(ngrams(question_wx, xgb_relevancy_shingle_len))

        vectorize_sample_x(X_data, 0, premise_shingles, question_shingles, xgb_relevancy_shingle2id)

        D_data = xgboost.DMatrix(X_data)
        y_probe = xgb_relevancy.predict(D_data)
        print(y_probe[0])

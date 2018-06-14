# -*- coding: utf-8 -*-
"""
Тренировка и валидация модели определения релевантности предпосылки и вопроса.
Модель используется в проекте чат-бота https://github.com/Koziev/chatbot
Движок - XGBoost. Для сравнения альтернативные модели: на базе LightGBM lgb_relevancy.py
и нейросетевая nn_relevancy.py

Программа содержит код для обучения (--run_mode train), интерактивной проверки
релевантности пар предложений, вводимых с консоли (--run_mode query) и
пакетной оценки качества на задаче выбора лучшей предпосылки для вопроса (--run_mode evaluate).

Предполагается, что датасет ../data/premise_question_relevancy.csv уже сгенерирован
программой prepare_relevancy_dataset.py

Пример запуска обучения с нужными параметрами командной строки см. в ../scripts/train_xgb_relevancy.sh

Обученная модель используется в чатботе в классе 
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

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

from utils.segmenter import Segmenter
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup
from utils.phrase_splitter import PhraseLemmatizer, PhraseTokenizer

config_filename = 'xgb_relevancy.config'

parser = argparse.ArgumentParser(description='XGB classifier for text relevance estimation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | evaluate | query | query2 | clusterize')
parser.add_argument('--shingle_len', type=int, default=3, help='shingle length')
parser.add_argument('--max_depth', type=int, default=6, help='"max_depth" parameter for XGBoost')
parser.add_argument('--eta', type=float, default=0.20, help='"eta" (learning rate) parameter for XGBoost')
parser.add_argument('--subsample', type=float, default=1.00, help='"subsample" parameter for XGBoost')
parser.add_argument('--lemmatize', type=int, default=0, help='lemmatize phrases before extracting the shingles')
parser.add_argument('--input', type=str, default='../data/premise_question_relevancy.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

args = parser.parse_args()

input_path = args.input
tmp_folder = args.tmp
data_folder = args.data_dir
run_mode = args.run_mode
lemmatize = args.lemmatize
subsample = args.subsample

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

# -------------------------------------------------------------------

if run_mode == 'train':
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    print('samples.count={}'.format(df.shape[0]))

    all_shingles = set()

    tokenizer = PhraseLemmatizer() if lemmatize else PhraseTokenizer()

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

    nb_0 = len(filter(lambda y: y == 0, y_data))
    nb_1 = len(filter(lambda y: y == 1, y_data))

    print('nb_0={}'.format(nb_0))
    print('nb_1={}'.format(nb_1))

    # ------------------------------------------------------------------------

    SEED = 123456
    TEST_SHARE = 0.2
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_data,
                                                                      y_data,
                                                                      weights,
                                                                      test_size=TEST_SHARE,
                                                                      random_state=SEED)

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

    model_filename = os.path.join( tmp_folder, 'xgb_relevancy.model' )

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'model': 'xgb',
                    'shingle2id': shingle2id,
                    'model_filename': model_filename,
                    'shingle_len': shingle_len,
                    'nb_features': nb_features,
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

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmatize = model_config['lemmatize']

    tokenizer = PhraseLemmatizer() if xgb_relevancy_lemmatize else PhraseTokenizer()

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

if run_mode == 'query2':
    # С клавиатуры задается путь к файлу с предложениями, и второе предложение.
    # Модель делает оценку релевантности для каждого предложения в файле и введенного предложения,
    # и сохраняет список оценок с сортировкой.

    # загружаем данные обученной модели
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmatize = model_config['lemmatize']

    tokenizer = PhraseLemmatizer() if xgb_relevancy_lemmatize else PhraseTokenizer()

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

            premise_shingles = set(ngrams(premise_wx, xgb_relevancy_shingle_len))
            question_shingles = set(ngrams(question_wx, xgb_relevancy_shingle_len))

            vectorize_sample_x(X_data, iphrase, premise_shingles, question_shingles, xgb_relevancy_shingle2id)

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

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmatize = model_config['lemmatize']

    tokenizer = PhraseLemmatizer() if xgb_relevancy_lemmatize else PhraseTokenizer()

    # Оценка качества натренированной модели на специальном наборе вопросов и ожидаемых выборов предпосылок
    # из тренировочного набора.
    eval_data = EvaluationDataset(0, tokenizer)
    eval_data.load(data_folder)

    xgb_relevancy = xgboost.Booster()
    xgb_relevancy.load_model(model_config['model_filename'])

    nb_good = 0  # попадание предпосылки в top-1
    nb_good5 = 0
    nb_good10 = 0
    nb_total = 0

    with codecs.open(os.path.join(tmp_folder, 'xgb_relevancy.evaluation.txt'), 'w', 'utf-8') as wrt:
        for irecord, phrases in eval_data.generate_groups():
            nb_samples = len(phrases)

            X_data = lil_matrix((nb_samples, xgb_relevancy_nb_features), dtype='bool')

            for irow, (premise_words, question_words) in enumerate(phrases):
                premise_wx = words2str(premise_words)
                question_wx = words2str(question_words)

                premise_shingles = set(ngrams(premise_wx, xgb_relevancy_shingle_len))
                question_shingles = set(ngrams(question_wx, xgb_relevancy_shingle_len))

                vectorize_sample_x(X_data, irow, premise_shingles, question_shingles, xgb_relevancy_shingle2id)

            D_data = xgboost.DMatrix(X_data)
            y_pred = xgb_relevancy.predict(D_data)

            # предпосылка с максимальной релевантностью
            max_index = np.argmax(y_pred)
            selected_premise = u' '.join(phrases[max_index][0]).strip()

            # эта выбранная предпосылка соответствует одному из вариантов
            # релевантных предпосылок в этой группе?
            nb_total += 1
            if eval_data.is_relevant_premise(irecord, selected_premise):
                nb_good += 1
                nb_good5 += 1
                nb_good10 += 1
                print(EvaluationMarkup.ok_color + EvaluationMarkup.ok_bullet + EvaluationMarkup.close_color, end='')
                wrt.write(EvaluationMarkup.ok_bullet)
            else:
                print(EvaluationMarkup.fail_color + EvaluationMarkup.fail_bullet + EvaluationMarkup.close_color, end='')
                wrt.write(EvaluationMarkup.fail_bullet)

                # среди top-5 или top-10 предпосылок есть верная?
                sorted_phrases = [x for x, _ in sorted(itertools.izip(phrases, y_pred), key=lambda z: -z[1])]

                for i in range(1, 10):
                    selected_premise = u' '.join(sorted_phrases[i][0]).strip()
                    if eval_data.is_relevant_premise(irecord, selected_premise):
                        if i < 5:
                            nb_good5 += 1  # верная предпосылка вошла в top-5
                        if i < 10:
                            nb_good10 += 1
                        break

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
    accuracy = float(nb_good)/float(nb_total)
    print('accuracy       ={}'.format(accuracy))

    # Также выведем точность попадания верной предпосылки в top-5 и top-10
    accuracy5 = float(nb_good5)/float(nb_total)
    print('accuracy top-5 ={}'.format(accuracy5))

    accuracy10 = float(nb_good10)/float(nb_total)
    print('accuracy top-10={}'.format(accuracy10))

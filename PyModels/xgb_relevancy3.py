# -*- coding: utf-8 -*-
"""
Эксперимент: тренировка модели определения релевантности предпосылки и вопроса
с помощью пережатых матриц соответствий шинглов в качестве фич.
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
import codecs

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

import skimage.transform

from utils.tokenizer import Tokenizer
from utils.segmenter import Segmenter
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup

config_filename = 'xgb_relevancy.config'

parser = argparse.ArgumentParser(description='XGB classifier for text relevance estimation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train evaluate query query2')
parser.add_argument('--shingle_len', type=int, default=3, help='shingle length')
parser.add_argument('--max_depth', type=int, default=6, help='max depth parameter for XGBoost')
parser.add_argument('--eta', type=float, default=0.20, help='eta (learning rate) parameter for XGBoost')
parser.add_argument('--input', type=str, default='../data/premise_question_relevancy.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

args = parser.parse_args()

input_path = args.input
tmp_folder = args.tmp
data_folder = args.data_dir
run_mode = args.run_mode

# основной настроечный параметр модели - длина символьных N-грамм (шинглов)
shingle_len = args.shingle_len

max_depth = args.max_depth
eta = args.eta

# -------------------------------------------------------------------

BEG_WORD = '\b'
END_WORD = '\n'


# размер изображения, которое получится после сжатия матрицы соответствия
# шинглов во входных предложениях.
shingle_image_size = 16


def shingles_list(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def get_shingle_image(str1, str2):
    shingles1 = shingles_list(str1, shingle_len)
    shingles2 = shingles_list(str2, shingle_len)
    image = np.zeros((len(shingles1), len(shingles2)), dtype='float32')
    for i1, shingle1 in enumerate(shingles1):
        for i2, shingle2 in enumerate(shingles2):
            if shingle1 == shingle2:
                image[i1, i2] = 1.0

    image_resized = skimage.transform.resize(image,
                                             (shingle_image_size, shingle_image_size))
    return image_resized.reshape(image_resized.size)



def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    return u' '.join(itertools.chain([BEG_WORD], words, [END_WORD]))


def vectorize_sample_x(X_data, idata, premise_str, question_str):
    X_data[idata, :] = get_shingle_image(premise_str, question_str)


# -------------------------------------------------------------------

tokenizer = Tokenizer()


if run_mode == 'train':
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    print('samples.count={}'.format(df.shape[0]))

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

    nb_features = shingle_image_size*shingle_image_size
    X_data = np.zeros((nb_patterns, nb_features), dtype='float32')
    y_data = []

    for idata, (phrase12, y12) in tqdm.tqdm(enumerate(itertools.izip(phrases, ys)), total=nb_patterns, desc='Vectorization'):
        premise_str = phrase12[0]
        question_str = phrase12[1]
        y = y12
        y_data.append(y)
        vectorize_sample_x( X_data, idata, premise_str, question_str )

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
                       num_boost_round=10000,
                       verbose_eval=50,
                       early_stopping_rounds=50)

    print('Training is finished')
    y_pred = cl.predict(D_val)
    y_pred = (y_pred >= 0.5).astype(np.int)
    acc = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred)
    print('val accuracy={}'.format(acc))

    f1 = sklearn.metrics.f1_score(y_true=y_val, y_pred=y_pred)
    print('val f1={}'.format(f1))

    model_filename = os.path.join( tmp_folder, 'xgb_relevancy3.model' )

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'model': 'xgb',
                    'model_filename': model_filename,
                    'nb_features': nb_features,
                    'shingle_image_size': shingle_image_size
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
    # Оценка качества натренированной модели на специальном наборе вопросов и ожидаемых выборов предпосылок
    # из тренировочного набора.
    eval_data = EvaluationDataset(0, tokenizer)
    eval_data.load(data_folder)

    # Загружаем данные обученной модели.
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_shingle_image_size = model_config['shingle_image_size']

    xgb_relevancy = xgboost.Booster()
    xgb_relevancy.load_model(model_config['model_filename'])

    nb_good = 0
    nb_bad = 0

    with codecs.open(os.path.join(tmp_folder, 'xgb_relevancy3.evaluation.txt'), 'w', 'utf-8') as wrt:
        for irecord, phrases in eval_data.generate_groups():
            nb_samples = len(phrases)

            X_data = np.zeros((nb_samples, xgb_relevancy_nb_features), dtype='float32')

            for irow, (premise_words, question_words) in enumerate(phrases):
                premise_wx = words2str(premise_words)
                question_wx = words2str(question_words)
                vectorize_sample_x(X_data, irow, premise_wx, question_wx)

            D_data = xgboost.DMatrix(X_data)
            y_pred = xgb_relevancy.predict(D_data)

            # предпосылка с максимальной релевантностью
            max_index = np.argmax(y_pred)
            selected_premise = u' '.join(phrases[max_index][0]).strip()

            # эта выбранная предпосылка соответствует одному из вариантов
            # релевантных предпосылок в этой группе?
            if eval_data.is_relevant_premise(irecord, selected_premise):
                nb_good += 1
                EvaluationMarkup.print_ok()
                wrt.write(EvaluationMarkup.ok_bullet)
            else:
                nb_bad += 1
                EvaluationMarkup.print_fail()
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

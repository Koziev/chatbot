# -*- coding: utf-8 -*-
"""
Тренировка модели классификации yes/no двух фраз (предпосылка и вопрос) для
вопросно-ответной системы https://github.com/Koziev/chatbot.

Используется XGBoost. Альтернативная нейросетевая реализация - см. nn_yes_no.py

Датасет должен быть сгенерирован и находится в папке ../data (см. prepare_qa_dataset.py)

19.03.2019 Добавлен gridsearch, в том числе подбор весов классов (scale_pos_weight)
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import codecs
import itertools
import json
import os
import argparse
import logging
import math

import numpy as np
import sklearn.metrics
import tqdm
import xgboost

from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from utils.tokenizer import Tokenizer
import utils.logging_helpers


# Основной гиперпараметр модели - число символов в N-граммах, мешок которых
# представляет анализируемое предложение.
SHINGLE_LEN = 3

BEG_WORD = '\b'
END_WORD = '\n'


class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer


def get_params_str(params):
    return u' '.join('{}={}'.format(p, v) for (p, v) in params.items())


def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    return u' '.join(itertools.chain([BEG_WORD], words, [END_WORD]))


def load_samples(input_path):
    tokenizer = Tokenizer()
    tokenizer.load()

    all_shingles = set(ngrams(words2str([]), SHINGLE_LEN))

    # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
    logging.info(u'Loading samples from {}'.format(input_path))
    samples0 = []
    nb_yes = 0  # кол-во ответов "да"
    nb_no = 0  # кол-во ответов "нет"
    max_nb_premises = 0  # макс. число предпосылок в сэмплах

    with codecs.open(input_path, 'r', 'utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if len(line) == 0:
                if len(lines) > 0:
                    premises = lines[:-2]
                    question = lines[-2]
                    answer = lines[-1]
                    if len(premises) <= 1:
                        sample = Sample(premises, question, answer)
                        samples0.append(sample)

                        max_nb_premises = max(max_nb_premises, len(premises))

                        if answer == u'да':
                            nb_yes += 1
                        elif answer == u'нет':
                            nb_no += 1

                        for phrase in lines:
                            words = tokenizer.tokenize(phrase)
                            wx = words2str(words)
                            #if u'меня ведь кеша зовут да' in wx:
                            #    pass

                            all_shingles.update(ngrams(wx, SHINGLE_LEN))

                    lines = []

            else:
                lines.append(line)

    logging.info('samples.count={}'.format(len(samples0)))
    logging.info('max_nb_premises={}'.format(max_nb_premises))

    nb_shingles = len(all_shingles)
    logging.info('nb_shingles={}'.format(nb_shingles))

    logging.info('nb_yes={}'.format(nb_yes))
    logging.info('nb_no={}'.format(nb_no))

    shingle2id = dict((s, i) for (i, s) in enumerate(all_shingles))

    # Сгенерируем внятные имена фич, чтобы увидеть алгоритм классификации в xgb
    feature_names = [u'common(' + shingle + u')' for (shingle, i) in sorted(shingle2id.iteritems(), key=lambda z: z[1])]
    feature_names.extend(
        [u'premise(' + shingle + u')' for (shingle, i) in sorted(shingle2id.iteritems(), key=lambda z: z[1])])
    feature_names.extend(
        [u'quest(' + shingle + u')' for (shingle, i) in sorted(shingle2id.iteritems(), key=lambda z: z[1])])

    nb_patterns = len(samples0)
    nb_features = nb_shingles * 3

    computed_params = {'shingle2id': shingle2id,
                       'feature_names': feature_names,
                       'nb_yes': nb_yes,
                       'nb_no': nb_no,
                       'nb_shingles': nb_shingles,
                       'nb_features': nb_features,
                       'max_nb_premises': max_nb_premises}

    X_data = lil_matrix((nb_patterns, nb_features), dtype='bool')
    y_data = np.zeros(nb_patterns, dtype='bool')
    samples = []
    idata = 0

    for index, sample in tqdm.tqdm(enumerate(samples0), total=nb_patterns, desc='Vectorization'):
        premise = sample.premises[0] if len(sample.premises) == 1 else u''
        question = sample.question
        answer = sample.answer

        y_data[idata] = answer == u'да'

        premise_words = tokenizer.tokenize(premise)
        question_words = tokenizer.tokenize(question)

        premise_shingles = ngrams(words2str(premise_words), SHINGLE_LEN)
        question_shingles = ngrams(words2str(question_words), SHINGLE_LEN)
        vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, shingle2id)

        samples.append((premise, question, answer))

        idata += 1

    #nb_0 = len(filter(lambda y: y == 0, y_data))
    #nb_1 = len(filter(lambda y: y == 1, y_data))

    return samples, X_data, y_data, computed_params


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
        if shingle not in shingle2id:
            pass
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_qs:
        X_data[idata, icol+shingle2id[shingle]] = True


def train_model(params, X_train, y_train, X_val, y_val):
    logging.info('Train model with params={}'.format(get_params_str(params)))
    D_train = xgboost.DMatrix(X_train, y_train, feature_names=computed_params['feature_names'], silent=0)
    D_val = xgboost.DMatrix(X_val, y_val, feature_names=computed_params['feature_names'], silent=0)

    xgb_params = {
        'booster': 'gbtree',
        'subsample': params['subsample'],
        'max_depth': params['max_depth'],
        'seed': 123456,
        'min_child_weight': params['min_child_weight'],
        'eta': params['eta'],
        'gamma': params['gamma'],
        'colsample_bytree': params['colsample_bytree'],
        'colsample_bylevel': params['colsample_bylevel'],
        'scale_pos_weight': params['scale_pos_weight'],
        'eval_metric': 'error',
        'objective': 'binary:logistic',
        'silent': 1,
        #'tree_method': 'gpu_hist',
        #'updater': 'grow_gpu',
    }

    cl = xgboost.train(xgb_params,
                       D_train,
                       evals=[(D_val, 'val')],
                       num_boost_round=5000,
                       verbose_eval=100,
                       early_stopping_rounds=50)

    logging.info('Training is finished')
    return cl


def score_model(cl, X_val, y_val):
    D_val = xgboost.DMatrix(X_val, y_val, feature_names=computed_params['feature_names'], silent=0)
    y_pred = cl.predict(D_val)
    y_pred = (y_pred >= 0.5).astype(np.int)
    f1 = sklearn.metrics.f1_score(y_true=y_val, y_pred=y_pred)
    return f1


def report_model(cl, samples_val, X_val, y_val, computed_params):
    D_val = xgboost.DMatrix(X_val, y_val, feature_names=computed_params['feature_names'], silent=0)

    y_pred = cl.predict(D_val)
    y_pred = (y_pred >= 0.5).astype(np.int)

    # Визуализация веса фич
    feature_scores = cl.get_fscore()
    with codecs.open('../tmp/feature_xgb_yes_no_scores.txt', 'w', 'utf-8') as wrt:
        for (feature, score) in sorted(feature_scores.iteritems(), key=lambda z: -z[1]):
            wrt.write(u'{}\t{}\n'.format(feature, score))

    # Результаты валидации сохраним в файле
    with codecs.open('../tmp/xgb_yes_no.validation.txt', 'w', 'utf-8') as wrt:
        for sample, y in itertools.izip(samples_val, y_pred):
            wrt.write(u'P: {}\n'.format(sample[0]))
            wrt.write(u'Q: {}\n'.format(sample[1]))
            wrt.write(u'A: {}\n'.format(sample[2]))
            wrt.write(u'y_pred={}\n\n'.format(y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost-based model for yes/no answer classification')
    parser.add_argument('--run_mode', type=str, default='train', choices='train gridsearch query'.split(),
                        help='what to do: train | query | gridsearch')
    parser.add_argument('--input', type=str, default='../data/pqa_yes_no.dat', help='path to input dataset')
    parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
    parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

    args = parser.parse_args()

    data_folder = args.data_dir
    input_path = args.input
    tmp_folder = args.tmp
    run_mode = args.run_mode

    utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'xgb_yes_no.log'))

    samples, X_data, y_data, computed_params = load_samples(input_path)

    if run_mode == 'gridsearch':
        params = dict()

        logging.info('Start gridsearch')

        best_params = None
        best_score = -np.inf
        crossval_count = 0

        for subsample in [1.0, 0.9, 0.8]:
            params['subsample'] = subsample

            for max_depth in [3, 4, 5]:
                params['max_depth'] = max_depth

                for min_child_weight in [1.0]:
                    params['min_child_weight'] = min_child_weight

                    for eta in [0.1, 0.2, 0.3]:
                        params['eta'] = eta

                        for gamma in [0.01, 0.1]:
                            params['gamma'] = gamma

                            for colsample_bytree in [1.0, 0.8]:
                                params['colsample_bytree'] = colsample_bytree

                                for colsample_bylevel in [1.0, 0.8]:
                                    params['colsample_bylevel'] = colsample_bylevel

                                    n0 = computed_params['nb_no']
                                    n1 = computed_params['nb_yes']
                                    for scale_pos_weight in [1.0, (n0/float(n1+n0)), math.sqrt((n0/float(n0+n1)))]:
                                        params['scale_pos_weight'] = scale_pos_weight

                                        crossval_count += 1
                                        logging.info('Start crossvalidation #{} for params={}'.format(crossval_count, get_params_str(params)))

                                        kf = StratifiedKFold(n_splits=3)
                                        scores = []
                                        for ifold, (train_index, val_index) in enumerate(kf.split(X_data, y_data)):
                                            logging.info('KFold[{}]'.format(ifold))

                                            X_train = X_data[train_index]
                                            y_train = y_data[train_index]
                                            X_val12 = X_data[val_index]
                                            y_val12 = y_data[val_index]

                                            X_val, X_finval, y_val, y_finval = train_test_split(X_val12,
                                                                                                y_val12,
                                                                                                test_size=0.5,
                                                                                                random_state=123456)

                                            cl = train_model(params, X_train, y_train, X_val, y_val)
                                            f1 = score_model(cl, X_finval, y_finval)
                                            logging.info('KFold[{}] f1={}'.format(ifold, f1))
                                            scores.append(f1)

                                        score = np.mean(scores)
                                        score_std = np.std(scores)
                                        logging.info('Crossvalidation #{} score={} std={}'.format(crossval_count, score, score_std))
                                        if score > best_score:
                                            best_params = params.copy()
                                            best_score = score
                                            logging.info('!!! NEW BEST score={} params={}'.format(best_score, get_params_str(best_params)))

        logging.info('Grid search complete, best_score={} best_params={}'.format(best_score, get_params_str(best_params)))

    if run_mode == 'train':
        logging.info('Train')

        params = dict()
        samples, X_data, y_data, computed_params = load_samples(input_path)

        params['subsample'] = 1.0
        params['max_depth'] = 5
        params['min_child_weight'] = 1.0
        params['eta'] = 0.30
        params['gamma'] = 0.1
        params['colsample_bytree'] = 1.0
        params['colsample_bylevel'] = 1.0
        params['scale_pos_weight'] = 1.0

        samples_train, samples_val, X_train, X_val, y_train, y_val = train_test_split(samples,
                                                                                      X_data,
                                                                                      y_data,
                                                                                      test_size=0.2,
                                                                                      random_state=123456)

        cl = train_model(params, X_train, y_train, X_val, y_val)

        # сохраним конфиг модели, чтобы ее использовать в чат-боте
        model_filename = os.path.join(tmp_folder, 'xgb_yes_no.model')
        model_config = {
                        'shingle2id': computed_params['shingle2id'],
                        'model_filename': model_filename,
                        'shingle_len': SHINGLE_LEN,
                        'nb_features': computed_params['nb_features'],
                        'feature_names': computed_params['feature_names']
                       }

        with open(os.path.join(tmp_folder, 'xgb_yes_no.config'), 'w') as f:
            json.dump(model_config, f, indent=4)

        cl.save_model(model_filename)

        f1 = score_model(cl, X_val, y_val)
        logging.info('Final model score f1={}'.format(f1))

        report_model(cl, samples_val, X_val, y_val, computed_params)

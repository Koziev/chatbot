# coding: utf-8
"""
Эксперименты с альтернативными моделями детектора перефразировок для чатбота.
Представление сопоставляемых фраз - мешок шинглов или фрагментов от SentencePiece.
Алгоритмы - различные бинарные классификаторы из sklearn.
Также проверяем влияние random projections.

Общий сценарий использования:
1) gridsearch - находим параметры оптимальной модели
2) eval - оценка оптимальной модели через кроссвалидацию
3) train - тренировка лучшей модели и сохранение ее в pickle файле
4) query - ручная проверка натренированной модели
5) hardnegative - аугментация датасета через подбор пар фраз, которые имеют высокую оценку релевантности


27-10-2019 переход на метрику mean reciprocal rank, перебор вариантов наборов фич
28-10-2019 добавлены классификаторы LightGBM и XGBoost
28-10-2019 добавлен метапараметр векторизации nlp_transform с лемматизацией
28-10-2019 из кода lgb_relevanvy перенесен сценарий "hardnegative" для автогенерации негативных сэмплов
"""

from __future__ import print_function
import pickle
import pandas as pd
import json
import os
import random
import logging
import logging.handlers
import numpy as np
import argparse
import collections
import io
import yaml
import tqdm

import scipy.sparse
from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import random_projection
from sklearn.model_selection import KFold

import xgboost
import lightgbm

#import sentencepiece as spm
import rutokenizer
import rupostagger
import rulemma

NFOLDS = 5

BEG_CHAR = u'\b'
END_CHAR = u'\n'


def get_params_str(model_params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in model_params.items())


def str2shingles(s0, n):
    s = BEG_CHAR + s0 + END_CHAR
    return [u''.join(z) for z in zip(*[s[i:] for i in range(n)])]


class ShingleVectorizer(object):
    def __init__(self, min_shingle_len, max_shingle_len, min_df):
        self.min_shingle_len = min_shingle_len
        self.max_shingle_len = max_shingle_len
        self.min_df = min_df
        self.shingle2index = dict()

    def fit(self, phrases):
        shingle2freq = collections.Counter()
        for phrase in phrases:
            for n in range(self.min_shingle_len, self.max_shingle_len+1):
                sx = str2shingles(phrase, n)
                shingle2freq.update(sx)

        good_shingles = (shingle for shingle, freq in shingle2freq.items() if freq >= self.min_df)
        self.shingle2index = dict((shingle, i) for (i, shingle) in enumerate(good_shingles))
        self.nb_shingles = len(self.shingle2index)

    def transform(self, phrases):
        res = []
        for phrase in phrases:
            phrase_sx = set()
            for n in range(self.min_shingle_len, self.max_shingle_len+1):
                phrase_sx.update(str2shingles(phrase, n))
            res.append(set(self.shingle2index[shingle] for shingle in phrase_sx if shingle in self.shingle2index))
        return res

    def get_feature_names(self):
        return [shingle for (shingle, index) in sorted(self.shingle2index.items(), key=lambda z: z[1])]


class SentencePieceVectorizer(object):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def fit(self, phrases):
        sp_corpus_path = os.path.join(tmp_dir, 'new_synonymy_detector.sentence_piece_corpus.txt')
        if not os.path.exists(sp_corpus_path):
            with io.open(sp_corpus_path, 'w', encoding='utf-8') as wrt:
                for phrase in phrases:
                    wrt.write(u'{}\n'.format(phrase))

        sp_model_name = 'new_synonymy_detector_{}'.format(self.vocab_size)
        if not os.path.exists(os.path.join(tmp_dir, sp_model_name + '.vocab')):
            logging.info('Start training SentencePiece for vocab_size={}'.format(self.vocab_size))
            spm.SentencePieceTrainer.Train(
                '--input={} --model_prefix={} --vocab_size={} --model_type=bpe'.format(sp_corpus_path, sp_model_name, self.vocab_size))
            os.rename(sp_model_name + '.vocab', os.path.join(tmp_dir, sp_model_name + '.vocab'))
            os.rename(sp_model_name + '.model', os.path.join(tmp_dir, sp_model_name + '.model'))

        self.splitter = spm.SentencePieceProcessor()
        self.splitter.Load(os.path.join(tmp_dir, sp_model_name + '.model'))

        pieces = set()
        for phrase in phrases:
            px = self.splitter.EncodeAsPieces(phrase)
            pieces.update(px)

        self.piece2index = dict((piece, i) for i, piece in enumerate(pieces))
        self.nb_shingles = len(self.piece2index)

    def transform(self, phrases):
        res = []
        for phrase in phrases:
            px = self.splitter.EncodeAsPieces(phrase)
            res.append(set(self.piece2index[piece] for piece in px if piece in self.piece2index))
        return res



def and_setlists(sets1, sets2):
    res = []
    for set1, set2 in zip(sets1, sets2):
        res.append(set1 & set2)
    return res


def or_setlists(sets1, sets2):
    res = []
    for set1, set2 in zip(sets1, sets2):
        res.append(set1 | set2)
    return res


def sub_setlists(sets1, sets2):
    res = []
    for set1, set2 in zip(sets1, sets2):
        res.append(set1 - set2)
    return res


def sets2matrix(sets, nb_shingles):
    X_data = lil_matrix((len(sets), nb_shingles), dtype='float32')
    for irow, row in enumerate(sets):
        for index in row:
            X_data[irow, index] = 1
    return X_data


vectorizers = dict()


def create_vectorizer(samples, params):
    phrases = list(s[0] for s in samples) + list(s[1] for s in samples)

    vectorizer_key = None
    if params['analyzer'] == 'char':
        vectorizer_key = '{}|{}|{}|{}|{}'.format(params['analyzer'], params['min_shingle_len'],
                                              params['max_shingle_len'], params['min_df'], params['featureset'])
    elif params['analyzer'] == 'sentencepiece':
        vectorizer_key = '{}|{}'.format(params['analyzer'], params['vocab_size'])
    else:
        raise NotImplementedError()

    if vectorizer_key in vectorizers:
        logging.info('Using already existing vectorizer for %s', vectorizer_key)
        vectorizer = vectorizers[vectorizer_key]
    else:
        logging.info('Creating new vectorizer for %s', vectorizer_key)
        if params['analyzer'] == 'char':
            vectorizer = ShingleVectorizer(params['min_shingle_len'],
                                           params['max_shingle_len'],
                                           params['min_df'])
        elif params['analyzer'] == 'sentencepiece':
            vectorizer = SentencePieceVectorizer(params['vocab_size'])
        else:
            raise NotImplementedError()

        vectorizer.fit(phrases)
        vectorizers[vectorizer_key] = vectorizer

    return vectorizer


def extract_lemma(token):
    return token[0] if token[1] == 'PRON' else token[2]


def lemmatize_phrase(phrase, tagger, lemmatizer):
    try:
        words = phrase.split()
        tags = tagger.tag(words)
        tokens = lemmatizer.lemmatize(tags)
        return u' '.join(map(extract_lemma, tokens))
    except Exception as ex:
        logging.error(u'Error occured in lemmatize_phrase for "%s"', phrase)
        logging.error(ex)
        exit(1)


def vectorize_data(samples, vectorizer, params):
    labels = [s[2] for s in samples]
    y_data = np.asarray(labels)

    phrases1 = [s[0] for s in samples]
    phrases2 = [s[1] for s in samples]

    if params['nlp_transform'] == 'lemmatize':
        tagger = rupostagger.RuPosTagger()
        tagger.load()

        lemmatizer = rulemma.Lemmatizer()
        lemmatizer.load()

        all_phrases = list(set(phrases1) | set(phrases2))
        phrase2lemma = dict((phrase, lemmatize_phrase(phrase, tagger, lemmatizer)) for phrase in all_phrases)
        lphrases1 = [phrase2lemma[f] for f in phrases1]
        lphrases2 = [phrase2lemma[f] for f in phrases2]
        return vectorize_data2(lphrases1, lphrases2, vectorizer, params), y_data
    else:
        return vectorize_data2(phrases1, phrases2, vectorizer, params), y_data


def vectorize_data2(phrases1, phrases2, vectorizer, params):
    ps = vectorizer.transform(phrases1)
    qs = vectorizer.transform(phrases2)

    nb_shingles = vectorizer.nb_shingles
    if params['random_proj'] == 0:
        common_shingles = sets2matrix(and_setlists(ps, qs), nb_shingles)
        notmatched_ps = sets2matrix(sub_setlists(ps, qs), nb_shingles)
        notmatched_qs = sets2matrix(sub_setlists(qs, ps), nb_shingles)

        if params['featureset'] == 0:
            X_data = scipy.sparse.hstack([common_shingles, notmatched_ps, notmatched_qs])
        elif params['featureset'] == 1:
            p_shingles = sets2matrix(ps, nb_shingles)
            q_shingles = sets2matrix(qs, nb_shingles)
            X_data = scipy.sparse.hstack([p_shingles, q_shingles, common_shingles, notmatched_ps, notmatched_qs])
        elif params['featureset'] == 2:
            different_shingles = sets2matrix(or_setlists(sub_setlists(ps, qs), sub_setlists(qs, ps)), nb_shingles)
            X_data = scipy.sparse.hstack([common_shingles, different_shingles])
        else:
            raise NotImplementedError()
    else:
        rp_transformer = random_projection.GaussianRandomProjection(100)
        #rp_transformer = random_projection.SparseRandomProjection()
        X1 = sets2matrix(ps, nb_shingles)
        X2 = sets2matrix(qs, nb_shingles)
        x12 = scipy.sparse.vstack((X1, X2))
        rp_transformer.fit(x12)
        X1 = rp_transformer.transform(X1)
        X2 = rp_transformer.transform(X2)
        X3 = np.subtract(X1, X2)
        X4 = np.multiply(X1, X2)
        X_data = np.hstack((X3, X4))
        print('X_data.shape={}'.format(X_data.shape))

    return X_data


def data_vectorization(samples, model_params):
    vectorizer = create_vectorizer(samples, model_params)
    X_data, y_data = vectorize_data(samples, vectorizer, model_params)
    return vectorizer, X_data, y_data


def create_estimator(model_params):
    if model_params['engine'] == 'SGDClassifier':
        cl = SGDClassifier(loss='hinge',
                           penalty=model_params['penalty'], alpha=0.0001, l1_ratio=0.15,
                           max_iter=10, tol=None, shuffle=True,
                           verbose=1, epsilon=0.1, n_jobs=4, random_state=None,
                           learning_rate='optimal', eta0=0.0, power_t=0.5,
                           # early_stopping=True, validation_fraction=0.1,
                           average=False, n_iter=None)
        return cl
    elif model_params['engine'] == 'LogisticRegression':
        cl = LogisticRegression(penalty=model_params['penalty'],
                                tol=0.0001,
                                C=model_params['C'],
                                verbose=0,
                                solver='saga',  # https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451
                                max_iter=100)
        return cl
    elif model_params['engine'] == 'LinearSVC':
        cl = LinearSVC(penalty=model_params['penalty'],
                       loss='squared_hinge',
                       dual=False,
                       tol=0.0001,
                       C=model_params['C'],
                       multi_class='ovr',
                       fit_intercept=True,
                       intercept_scaling=1,
                       class_weight=None,
                       verbose=0,
                       random_state=None,
                       max_iter=2000)
        return cl
    elif model_params['engine'] == 'SVC':
        cl = SVC(#penalty=model_params['penalty'],
                 #loss='squared_hinge',
                 #dual=False,
                 tol=0.0001,
                 C=model_params['C'],
                 #multi_class='ovr',
                 #fit_intercept=True,
                 #intercept_scaling=1,
                 class_weight=None,
                 verbose=0,
                 random_state=None,
                 max_iter=2000)
        return cl
    elif model_params['engine'] == 'GBM':
        cl = GradientBoostingClassifier(loss='deviance',
                                        learning_rate=model_params['learning_rate'],
                                        n_estimators=model_params['n_estimators'],
                                        subsample=model_params['subsample'],
                                        criterion='friedman_mse',
                                        min_samples_split=model_params['min_samples_split'],
                                        min_samples_leaf=model_params['min_samples_leaf'],
                                        min_weight_fraction_leaf=0.0,
                                        max_depth=model_params['max_depth'],
                                        #min_impurity_split=1e-07,
                                        init=None,
                                        random_state=1234567,
                                        #max_features=None,
                                        verbose=0,
                                        max_leaf_nodes=None, warm_start=False, presort='auto')
        return cl
    elif model_params['engine'] == "XGB":
        cl = xgboost.XGBClassifier(max_depth=int(model_params['max_depth']),
                                   learning_rate=float(model_params['learning_rate']),
                                   n_estimators=int(model_params['n_estimators']),
                                   verbosity=1, silent=1,
                                   objective="binary:logistic", booster='gbtree',
                                   n_jobs=8,
                                   gamma=0, min_child_weight=1,
                                   max_delta_step=0, subsample=float(model_params['subsample']),
                                   colsample_bytree=1,
                                   colsample_bylevel=1, colsample_bynode=1,
                                   reg_alpha=0, reg_lambda=1,
                                   scale_pos_weight=1,
                                   random_state=31415926)
        return cl
    elif model_params['engine'] == "LGB":
        cl = lightgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=int(model_params['num_leaves']),
                                     max_depth=-1, learning_rate=float(model_params['learning_rate']),
                                     n_estimators=int(model_params['n_estimators']),
                                     subsample_for_bin=200000, objective="binary",
                                     class_weight=None, min_split_gain=0.0,
                                     min_child_weight=0.001, min_child_samples=20,
                                     subsample=float(model_params['subsample']),
                                     subsample_freq=0,
                                     colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None,
                                     n_jobs=8,
                                     silent=True, importance_type='split')
        return cl
    else:
        raise NotImplementedError('engine="{}" is not implemented'.format(model_params['engine']))


class GridGenerator(object):
    def __init__(self):
        pass

    def vectorizer_grid(self):
        for analyzer in ['char']:  # 'sentencepiece', 'char'
            if analyzer == 'sentencepiece':
                for vocab_size in [4000, 8000]:
                    for random_proj in [0]:
                        model_params = dict()
                        model_params['analyzer'] = analyzer
                        model_params['vocab_size'] = vocab_size
                        model_params['random_proj'] = random_proj
                        yield model_params
            else:
                for nlp_transform in ['lemmatize', '']:  # 'lemmatize',
                    for random_proj in [0]:
                        for min_shingle_len in [3]:
                            for max_shingle_len in [4]:
                                for min_df in [1]:
                                    for featureset in [2, 0, 1]:
                                        model_params = dict()
                                        model_params['nlp_transform'] = nlp_transform
                                        model_params['analyzer'] = analyzer
                                        model_params['random_proj'] = random_proj
                                        model_params['min_shingle_len'] = min_shingle_len
                                        model_params['max_shingle_len'] = max_shingle_len
                                        model_params['min_df'] = min_df
                                        model_params['featureset'] = featureset
                                        yield model_params

    def estimator_grid(self):
        for engine in ['LGB']:  # 'LogisticRegression', 'GBM', 'LGB', 'XGB'
            if engine == 'GBM':
                for learning_rate in [0.1, 0.2]:
                    for n_estimators in [500, 1000]:
                        for subsample in [1.0]:
                            for min_samples_split in [2]:
                                for min_samples_leaf in [1]:
                                    for max_depth in [5, 6]:
                                        model_params = dict()
                                        model_params['engine'] = engine
                                        model_params['max_depth'] = max_depth
                                        model_params['learning_rate'] = learning_rate
                                        model_params['n_estimators'] = n_estimators
                                        model_params['subsample'] = subsample
                                        model_params['min_samples_split'] = min_samples_split
                                        model_params['min_samples_leaf'] = min_samples_leaf
                                        yield model_params

            elif engine == 'XGB':
                for learning_rate in [0.2, 0.3]:
                    for n_estimators in [400, 1000, 1500]:
                        for subsample in [1.00]:
                            for max_depth in [5, 6, 7, 8]:
                                model_params = dict()
                                model_params['engine'] = engine
                                model_params['learning_rate'] = learning_rate
                                model_params['n_estimators'] = n_estimators
                                model_params['subsample'] = subsample
                                model_params['max_depth'] = max_depth
                                yield model_params

            elif engine == 'LGB':
                for learning_rate in [0.25, 0.3, 0.35]:
                    for n_estimators in [500, 1000, 1500]:
                        for subsample in [1.0]:
                            for num_leaves in [31, 50, 100]:
                                model_params = dict()
                                model_params['engine'] = engine
                                model_params['learning_rate'] = learning_rate
                                model_params['n_estimators'] = n_estimators
                                model_params['subsample'] = subsample
                                model_params['num_leaves'] = num_leaves
                                yield model_params

            else:
                for penalty in ['l2']:
                    for model_C in [1e4, 1e5, 1e6]:
                        model_params = dict()
                        model_params['engine'] = engine
                        model_params['penalty'] = penalty
                        model_params['C'] = model_C
                        yield model_params


def compute_ranking_accuracy(estimator, vectorizer, model_params, val_samples):
    logging.debug('ENTER compute_ranking_accuracy')
    # Код для получения оценочных метрик качества ранжирования.
    premise2samples = dict()

    # Опорные сэмплы из релевантных пар
    for sample in val_samples:
        if sample[2] == 1:
            phrase1 = sample[0]
            phrase2 = sample[1]
            if phrase1 not in premise2samples:
                premise2samples[phrase1] = [(phrase2, 1)]

    premise2samples = dict(random.sample(premise2samples.items(), 1000))
    logging.debug('premise2samples.count={}'.format(len(premise2samples)))

    # Добавляем заданные в датасете нерелевантные пары
    for sample in val_samples:
        if sample[2] == 0:
            phrase1 = sample[0]
            phrase2 = sample[1]
            if phrase1 in premise2samples:
                premise2samples[phrase1].append((phrase2, 0))

    # Теперь в каждую группу добавим рандомных случайных сэмплов.
    group_size = 100
    for phrase1_1, samples_1 in premise2samples.items():
        for sample in sorted(val_samples, key=lambda _: random.random()):
            phrase1 = sample[0]
            phrase2 = sample[1]

            # Добавляем вторую фразу как нерелевантный сэмпл к каждому левому предложению phrase1_1.
            samples3 = []
            all_samples = set(samples_1)
            if phrase1 != phrase1_1:
                if (phrase2, 0) not in all_samples and (phrase2, 1) not in all_samples:
                    samples3.append((phrase2, 0))
                    all_samples.add((phrase2, 0))

            if len(samples3) >= group_size*5:
                break

        samples3 = list(samples3)
        samples3 = samples3[:group_size-len(samples_1)]
        premise2samples[phrase1_1].extend(samples3)

    # Теперь в premise2samples для каждой фразы-ключа есть некоторое количество (group_size)
    # сравниваемых фраз, из которых только 1 релевантна. Мы должны проверить, что модель
    # именно эту пару оценит максимально высоко, а остальным присвоит меньшую релевантность.
    nb_good = 0
    nb_total = 0

    logging.debug('Processing premise2samples...')
    ranks = []
    for phrase1, samples2 in premise2samples.items():
        samples3 = [(phrase1, s[0], s[1]) for s in samples2]
        X_data, y_data = vectorize_data(samples3, vectorizer, model_params)
        y_pred = estimator.predict_proba(X_data)[:, 1]
        maxy_pred = np.argmax(y_pred)
        maxy_data = np.argmax(y_data)
        nb_good += int(maxy_pred == maxy_data)
        nb_total += 1

        phrase_y = [(s, y) for s, y in zip(samples2, y_pred)]
        phrase_y = sorted(phrase_y, key=lambda z: -z[1])

        # Ищем позицию сэмпла с релевантной парой - это будет ее ранг (прибавить 1!).
        true_pos = next(i for (i, z) in enumerate(phrase_y) if z[0][1] == 1)
        ranks.append(true_pos)

    # mean reciprocal rank
    # https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    mrr = np.mean([1./(1.0+rank) for rank in ranks])

    # precision@1
    # https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K
    precision1 = float(nb_good) / nb_total

    logging.debug('precision@1=%f mrr=%f', precision1, mrr)

    return precision1, mrr


def collect_strings(d):
    res = []

    if isinstance(d, str):
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


def print_shingle(shingle):
    return shingle.replace(BEG_CHAR, r'\b').replace(END_CHAR, r'\n')


def get_feature_names(vectorizer, model_params):
    shingles = vectorizer.get_feature_names()
    feature_names = []

    if model_params['featureset'] == 0:
        for shingle in shingles:
            feature_names.append(u'{}(a&b)'.format(print_shingle(shingle)))

        for shingle in shingles:
            feature_names.append(u'{}(a-b)'.format(print_shingle(shingle)))

        for shingle in shingles:
            feature_names.append(u'{}(b-a)'.format(print_shingle(shingle)))

    elif model_params['featureset'] == 1:
        for shingle in shingles:
            feature_names.append(u'{}(a)'.format(print_shingle(shingle)))

        for shingle in shingles:
            feature_names.append(u'{}(b)'.format(print_shingle(shingle)))

        for shingle in shingles:
            feature_names.append(u'{}(a&b)'.format(print_shingle(shingle)))

        for shingle in shingles:
            feature_names.append(u'{}(a-b)'.format(print_shingle(shingle)))

        for shingle in shingles:
            feature_names.append(u'{}(b-a)'.format(print_shingle(shingle)))
    elif model_params['featureset'] == 2:
        for shingle in shingles:
            feature_names.append(u'{}(a&b)'.format(print_shingle(shingle)))

        for shingle in shingles:
            feature_names.append(u'{}(a<>b)'.format(print_shingle(shingle)))
    else:
        raise NotImplementedError()

    return feature_names


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path, encoding='utf-8', delimiter='\t', index_col=None, keep_default_na=False)
    samples = []
    for phrase1, phrase2, label in zip(df['premise'].values, df['question'].values, df['relevance'].values):
        samples.append((phrase1.strip(), phrase2.strip(), label))
    return samples


def flush_logging():
    for h in logging._handlerList:
        h().flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synonymy/paraphrase detector training and evaluation kit')
    parser.add_argument('--run_mode', choices='gridsearch eval train query query2 hardnegative'.split(), default='gridsearch')
    parser.add_argument('--tmp_dir', default='../../tmp')
    parser.add_argument('--dataset', default='../../data/synonymy_dataset.csv')
    args = parser.parse_args()

    tmp_dir = args.tmp_dir
    dataset_path = args.dataset
    run_mode = args.run_mode

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    lf = logging.FileHandler(os.path.join(tmp_dir, 'new_synonymy_detector.log'), mode='w')
    lf.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)

    logging.info('Start using dataset_path="%s"', dataset_path)

    # файл в формате json с найденными оптимальными параметрами классификатора,
    # создается в ходе gridsearch, используется для train
    best_params_path = os.path.join(tmp_dir, 'new_synonymy_detector.best_params.json')

    # файл с обученной моделью, создается в train, используется в query
    model_path = os.path.join(tmp_dir, 'new_synonymy_detector.model')

    if run_mode == 'gridsearch':
        logging.info('=== GRIDSEARCH ===')

        # На полном датасете подбор идет слишком долго, ограничим.
        samples = load_dataset(dataset_path)
        samples = random.sample(samples, 10000)

        best_model_params = None
        best_score = 0.0
        crossval_count = 0

        grid = GridGenerator()
        for vectorizer_params in grid.vectorizer_grid():
            vectorizer = create_vectorizer(samples, vectorizer_params)
            for estimator_params in grid.estimator_grid():
                model_params = vectorizer_params.copy()
                model_params.update(estimator_params)
                logging.info(u'Cross-validation using model_params: {}'.format(get_params_str(model_params)))

                crossval_count += 1
                kf = KFold(n_splits=NFOLDS)
                scores = []
                mrrs = []
                for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
                    train_samples = [samples[i] for i in train_index]
                    val_samples = [samples[i] for i in val_index]

                    X_train, y_train = vectorize_data(train_samples, vectorizer, model_params)
                    X_val, y_val = vectorize_data(val_samples, vectorizer, model_params)

                    estimator = create_estimator(model_params)
                    estimator.fit(X_train, y_train)
                    precision1, mrr = compute_ranking_accuracy(estimator, vectorizer, model_params, val_samples)
                    scores.append(precision1)
                    mrrs.append(mrr)
                    logging.info('fold %d/%d precision@1=%g mrr=%g', ifold+1, NFOLDS, precision1, mrr)

                precision1 = np.mean(scores)
                precision1_std = np.std(scores)
                logging.info('Crossvalidation #{} precision@1={} std={} mean reciprocal rank={}'.format(crossval_count, precision1, precision1_std, np.mean(mrrs)))

                if precision1 > best_score:
                    logging.info('!!! NEW BEST !!! precision@1={} params={}'.format(precision1, get_params_str(model_params)))
                    best_score = precision1
                    best_model_params = model_params.copy()
                    with open(best_params_path, 'w') as f:
                        json.dump(best_model_params, f, indent=4)
                else:
                    logging.info('No improvement over current best_score={}'.format(best_score))

        logging.info('best_score={} for model_params: {}'.format(best_score, get_params_str(best_model_params)))

    if run_mode == 'eval':
        # Оценка лучшей модели через кроссвалидацию на полном датасете.
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        samples = load_dataset(dataset_path)
        vectorizer = create_vectorizer(samples, best_model_params)

        kf = KFold(n_splits=NFOLDS)
        scores = []
        mrrs = []
        for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
            print('KFold[{}]'.format(ifold))
            train_samples = [samples[i] for i in train_index]
            val_samples = [samples[i] for i in val_index]

            X_train, y_train = vectorize_data(train_samples, vectorizer, best_model_params)
            X_val, y_val = vectorize_data(val_samples, vectorizer, best_model_params)

            estimator = create_estimator(best_model_params)
            estimator.fit(X_train, y_train)
            precision1, mrr = compute_ranking_accuracy(estimator, vectorizer, best_model_params, val_samples)
            scores.append(precision1)
            mrrs.append(mrr)

        precision1 = np.mean(scores)
        score_std = np.std(scores)
        logging.info('Cross-validation precision@1=%f std=%f mrr=%f', precision1, score_std, np.mean(mrrs))

    if run_mode == 'train':
        # Тренировка финальной модели на полном датасете. Используются метапараметры, найденные
        # в ходе gridsearch.

        logging.info('Loading best_model_params from "%s"', best_params_path)
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        samples = load_dataset(dataset_path)
        best_vectorizer, X_data, y_data = data_vectorization(samples, best_model_params)
        logging.info('Train on %d samples', len(samples))

        # финальное обучение классификатора на всех данных
        logging.info('Training the final classifier model_params: %s', get_params_str(best_model_params))
        estimator = create_estimator(best_model_params)
        estimator.fit(X_data, y_data)

        # сохраним натренированный классификатор и дополнительные параметры, необходимые для
        # использования модели в чатботе.
        model = {'vectorizer': best_vectorizer,
                 'estimator': estimator,
                 'model_params': best_model_params}

        logging.info('Storing model to "%s"', model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Выведем результаты предсказания обучающего датасета - для отладки и визуального контроля, чтобы
        # потом легче было проверять работу сценариев query, query2 и т.д.
        pred_filepath = os.path.join(tmp_dir, 'new_synonymy_detector.train_prediction.txt')
        logging.info('Writing predictons to "%s"', pred_filepath)
        with io.open(pred_filepath, 'w', encoding='utf-8') as wrt:
            samples2 = samples  #random.sample(samples, 1000)
            X_data, y_data = vectorize_data(samples2, best_vectorizer, best_model_params)
            y_pred = estimator.predict_proba(X_data)[:, 1]
            for sample, y_pred1, y_true1 in sorted(zip(samples2, y_pred, y_data), key=lambda z: z[1]):
                wrt.write(u'{}\n'.format(sample[0]))
                wrt.write(u'{}\n'.format(sample[1]))
                wrt.write(u'y_true={} y_pred={}\n'.format(y_true1, y_pred1))
                wrt.write(u'\n\n')

        # Выведем веса признаков для визуального анализа
        feature_names = get_feature_names(best_vectorizer, best_model_params)
        with io.open(os.path.join(tmp_dir, 'new_synonymy_detector.features.txt'), 'w', encoding='utf-8') as wrt:
            feature_weights = []
            if isinstance(estimator, LogisticRegression) or isinstance(estimator, LinearSVC):
                assert len(estimator.coef_) == 1
                nb_features = len(feature_names)
                for feature_index in range(nb_features):
                    feature_term = feature_names[feature_index]
                    feature_weight = estimator.coef_[0][feature_index]
                    if feature_weight != 0.0:
                        feature_weights.append((feature_term, feature_weight))

            # Выведем отдельно топ-100 негативных и позитивных
            for feature, weight in sorted(feature_weights, key=lambda z: -z[1])[:100]:
                wrt.write(u'{:<10s} = {}\n'.format(feature, weight))
            wrt.write('\n\n...\n\n\n')
            for feature, weight in sorted(feature_weights, key=lambda z: -z[1])[-100:]:
                wrt.write(u'{:<10s} = {}\n'.format(feature, weight))

        logging.info('All done.')

    if run_mode == 'query':
        logging.info('Restoring model from "%s"', model_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        flush_logging()

        while True:
            phrase1 = input('1:> ').strip().lower()
            phrase2 = input('2:> ').strip().lower()

            samples = [(phrase1, phrase2, 0)]
            X_data, y_data = vectorize_data(samples, model['vectorizer'], model['model_params'])
            y_query = model['estimator'].predict_proba(X_data)[:, 1]
            y = y_query[0]
            print(u'{}'.format(y))

    if run_mode == 'query2':
        data_folder = '../../data'
        tokenizer = rutokenizer.Tokenizer()
        tokenizer.load()

        # Проверка модели через поиск ближайших сэмплов в большом наборе предложений.
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        logging.info('Restoring model from "%s"', model_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        vectorizer = model['vectorizer']

        premises = []
        added_phrases = set()

        # поиск ближайшего приказа или вопроса из списка FAQ
        phrases2 = set()
        if True:
            samples = load_dataset(dataset_path)
            for sample in samples:
                for phrase in [sample[0], sample[1]]:
                    if phrase not in added_phrases:
                        added_phrases.add(phrase)
                        phrases2.add((phrase, phrase))

        if False:
            for phrase in load_strings_from_yaml(os.path.join(data_folder, 'rules.yaml')):
                phrase2 = u' '.join(tokenizer.tokenize(phrase))
                if '|' not in phrase2:
                    if phrase2 not in added_phrases:
                        added_phrases.add(phrase2)
                        phrases2.add((phrase2, phrase))

        if False:
            with io.open(os.path.join(data_folder, 'intents.txt'), 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    phrase = line.strip()
                    if len(phrase) > 5 and not phrase.startswith('#') and u'_' not in phrase:
                        phrase2 = u' '.join(tokenizer.tokenize(phrase))
                        if phrase2 not in added_phrases:
                            added_phrases.add(phrase2)
                            phrases2.add((phrase2, phrase))

        if False:
            with io.open(os.path.join(data_folder, 'faq2.txt'), 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    phrase = line.strip()
                    if len(phrase) > 5 and phrase.startswith(u'Q:'):
                        phrase = phrase.replace(u'Q:', u'').strip()
                        phrase2 = u' '.join(tokenizer.tokenize(phrase))
                        if phrase2 not in added_phrases:
                            added_phrases.add(phrase2)
                            phrases2.add((phrase2, phrase))

        phrases = list(phrases2)
        nb_phrases = len(phrases2)
        logging.info("phrases.count=%d", nb_phrases)
        flush_logging()

        while True:
            phrase1 = input(':> ').strip().lower()
            phrase1 = u' '.join(tokenizer.tokenize(phrase1))

            samples = [(phrase1, phrase2[1], 0) for phrase2 in phrases2]
            X_data, _ = vectorize_data(samples, vectorizer, best_model_params)
            y_query = model['estimator'].predict_proba(X_data)[:, 1]

            phrase_w = [(phrases[i][1], y_query[i]) for i in range(nb_phrases)]
            for phrase2, score in sorted(phrase_w, key=lambda z: -z[1])[:20]:
                print(u'{:8.5f}\t{}'.format(score, phrase2))

    if run_mode == 'hardnegative':
        # Поиск новых негативных сэмплов, которые надо добавить в датасет для
        # уменьшения количества неверно определяемых положительных пар.
        # Алгоритм: для сэмпла из ручного датасета определяем релевантность к остальным
        # репликам в этом же датасете. Отбираем те реплики, которые дают оценку
        # релевантности >0.5, исключаем правильные положительные и известные негативные,
        # остаются те фразы, которые считаются релевантными исходной фразе, но это неверно.
        # Сохраняем получающийся список в файле для ручной модерации.
        data_folder = os.path.dirname(dataset_path)
        tokenizer = rutokenizer.Tokenizer()
        tokenizer.load()

        logging.info('Restoring model from "%s"', model_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        known_pairs = set()
        test_phrases = set()
        if True:  #task == 'synonymy':
            if True:
                with io.open(os.path.join(data_folder, 'paraphrases.txt'), 'r', encoding='utf-8') as rdr:
                    block = []
                    for line in rdr:
                        phrase = line.replace('(-)', '').replace('(+)', '').strip()
                        if len(phrase) == 0:
                            for phrase1 in block:
                                for phrase2 in block:
                                    known_pairs.add((phrase1, phrase2))
                            block = []
                        else:
                            if len(phrase) > 5 and not phrase.startswith('#') and u'_' not in phrase:
                                words = tokenizer.tokenize(phrase)
                                if len(words) > 2:
                                    phrase2 = u' '.join(words)
                                    test_phrases.add((phrase2, phrase))
                                    block.append(phrase)

            if True:
                with io.open(os.path.join(data_folder, 'intents.txt'), 'r', encoding='utf-8') as rdr:
                    for line in rdr:
                        phrase = line.strip()
                        if len(phrase) > 5 and not phrase.startswith('#') and u'_' not in phrase:
                            phrase2 = u' '.join(tokenizer.tokenize(phrase))
                            test_phrases.add((phrase2, phrase))

            if True:
                with io.open(os.path.join(data_folder, 'faq2.txt'), 'r', encoding='utf-8') as rdr:
                    for line in rdr:
                        phrase = line.strip()
                        if len(phrase) > 5 and phrase.startswith(u'Q:'):
                            phrase = phrase.replace(u'Q:', u'').strip()
                            words = tokenizer.tokenize(phrase)
                            if len(words) > 2:
                                phrase2 = u' '.join(words)
                                test_phrases.add((phrase2, phrase))
        else:
            raise NotImplementedError()

        premises = list(test_phrases)
        nb_premises = len(premises)
        logging.info('nb_premises=%d', nb_premises)

        with io.open(os.path.join(tmp_dir, 'new_synonymy_detector.hard_negatives.txt'), 'w', encoding='utf-8') as wrt:
            nb_stored = 0  # кол-во найденных и сохраненных негативных примеров
            for iphrase, (nphrase, question) in enumerate(sorted(premises, key=lambda _: random.random())):
                samples2 = [(nphrase, f[1], 0) for f in test_phrases]
                X_data, y_data = vectorize_data(samples2, model['vectorizer'], model['model_params'])
                y_pred = model['estimator'].predict(X_data)

                selected_phrases2 = []
                phrase_rels = [(premises[i][1], y_pred[i]) for i in range(nb_premises) if y_pred[i] > 0.5]
                phrase_rels = sorted(phrase_rels, key=lambda z: -z[1])
                for phrase2, sim in phrase_rels[:20]:
                    if phrase2 != question and (question, phrase2) not in known_pairs and (
                    phrase2, question) not in known_pairs:
                        selected_phrases2.append(phrase2)
                if len(selected_phrases2) > 0:
                    wrt.write(u'{}\n'.format(question))
                    for phrase2 in selected_phrases2:
                        wrt.write(u'(-) {}\n'.format(phrase2))
                    wrt.write(u'\n\n')
                    wrt.flush()
                    nb_stored += len(selected_phrases2)
                    logging.info('%d/%d processed, %d negative samples stored', iphrase, nb_premises, nb_stored)

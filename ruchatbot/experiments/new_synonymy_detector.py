# coding: utf-8
"""
Эксперименты с альтернативными моделями детектора перефразировок.
Представление сопоставляемых фраз - мешок шинглов или фрагментов от SentencePiece.
Алгоритмы - различные бинарные классификаторы из sklearn.
Также проверяем влияние random projections.

Общий сценарий использования:
1) gridsearch - находим параметры оптимальной модели
2) eval - оценка оптимальной модели через кроссвалидацию
3) train - тренировка лучшей модели и сохранение ее в pickle файле
4) query - ручная проверка натренированной модели
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
import itertools
import collections
import io
import yaml

import scipy.sparse
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import random_projection
from sklearn.model_selection import KFold

#import sentencepiece as spm
import rutokenizer

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
        vectorizer_key = '{}|{}|{}|{}'.format(params['analyzer'], params['min_shingle_len'],
                                              params['max_shingle_len'], params['min_df'])
    elif params['analyzer'] == 'sentencepiece':
        vectorizer_key = '{}|{}'.format(params['analyzer'], params['vocab_size'])
    else:
        raise NotImplementedError()

    if vectorizer_key in vectorizers:
        logging.info('Using already existing vectorizer for {}'.format(vectorizer_key))
        vectorizer = vectorizers[vectorizer_key]
    else:
        logging.info('Creating new vectorizer for {}'.format(vectorizer_key))
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


def vectorize_data(samples, vectorizer, params):
    labels = [s[2] for s in samples]
    y_data = np.asarray(labels)

    phrases1 = [s[0] for s in samples]
    phrases2 = [s[1] for s in samples]

    return vectorize_data2(phrases1, phrases2, vectorizer, params), y_data


def vectorize_data2(phrases1, phrases2, vectorizer, params):
    ps = vectorizer.transform(phrases1)
    qs = vectorizer.transform(phrases2)

    nb_shingles = vectorizer.nb_shingles
    if params['random_proj'] == 0:
        common_shingles = sets2matrix(and_setlists(ps, qs), nb_shingles)
        notmatched_ps = sets2matrix(sub_setlists(ps, qs), nb_shingles)
        notmatched_qs = sets2matrix(sub_setlists(qs, ps), nb_shingles)
        X_data = scipy.sparse.hstack([common_shingles, notmatched_ps, notmatched_qs])
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


def data_vectorization(df, model_params):
    vectorizer = create_vectorizer(df, model_params)
    X_data, y_data = vectorize_data(df, vectorizer, model_params)
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
                                n_jobs=4,
                                multi_class='multinomial', verbose=0, solver='lbfgs',
                                max_iter=1000)
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
    else:
        raise NotImplementedError('Model engine={} is not implemented'.format(model_params['engine']))


class GridGenerator(object):
    def __init__(self):
        pass

    def generate(self):
        model_params = dict()

        for analyzer in ['char']:  # 'sentencepiece', 'char'
            model_params['analyzer'] = analyzer

            if analyzer == 'sentencepiece':
                for vocab_size in [4000, 8000]:
                    model_params['vocab_size'] = vocab_size

                    for random_proj in [0]:
                        model_params['random_proj'] = random_proj

                        for engine in ['LogisticRegression']:  #, 'GBM', 'LogisticRegression', 'SVC']:
                            model_params['engine'] = engine

                            if engine == 'GBM':
                                for learning_rate in [0.05, 0.1, 0.3]:
                                    model_params['learning_rate'] = learning_rate

                                    for n_estimators in [50, 100, 200]:
                                        model_params['n_estimators'] = n_estimators

                                        for subsample in [1.0, 0.8]:
                                            model_params['subsample'] = subsample

                                            for min_samples_split in [2]:
                                                model_params['min_samples_split'] = min_samples_split

                                                for min_samples_leaf in [1]:
                                                    model_params['min_samples_leaf'] = min_samples_leaf

                                                    for max_depth in [3, 4, 5]:
                                                        model_params['max_depth'] = max_depth

                                                        yield model_params
                            else:
                                for penalty in ['l2']:
                                    model_params['penalty'] = penalty
                                    for model_C in [1e2, 1e3, 1e4, 1e5]:
                                        model_params['C'] = model_C
                                        yield model_params
            else:
                for random_proj in [0]:
                    model_params['random_proj'] = random_proj

                    for min_shingle_len in [3]:
                        model_params['min_shingle_len'] = min_shingle_len
                        for max_shingle_len in [4]:
                            model_params['max_shingle_len'] = max_shingle_len
                            for min_df in [1]:
                                model_params['min_df'] = min_df

                                for engine in ['LogisticRegression']:  # 'LinearSVC', 'GBM', 'LogisticRegression', 'SVC'
                                    model_params['engine'] = engine

                                    if engine == 'GBM':
                                        for learning_rate in [0.05, 0.1, 0.3]:
                                            model_params['learning_rate'] = learning_rate

                                            for n_estimators in [50, 100, 200]:
                                                model_params['n_estimators'] = n_estimators

                                                for subsample in [1.0, 0.8]:
                                                    model_params['subsample'] = subsample

                                                    for min_samples_split in [2]:
                                                        model_params['min_samples_split'] = min_samples_split

                                                        for min_samples_leaf in [1]:
                                                            model_params['min_samples_leaf'] = min_samples_leaf

                                                            for max_depth in [3, 4, 5]:
                                                                model_params['max_depth'] = max_depth

                                                                yield model_params
                                    else:
                                        for penalty in ['l2']:
                                            model_params['penalty'] = penalty
                                            for model_C in [1e2, 1e3, 1e4, 1e5]:
                                                model_params['C'] = model_C
                                                yield model_params


def compute_ranking_accuracy(estimator, vectorizer, model_params, val_samples):
    # Код для получения оценочной метрики "качество ранжирования".
    premise2samples = dict()

    # Опорные сэмплы из релевантных пар
    for sample in val_samples:
        if sample[2] == 1:
            phrase1 = sample[0]
            phrase2 = sample[1]
            if phrase1 not in premise2samples:
                premise2samples[phrase1] = [(phrase2, 1)]

    # Добавляем заданные в датасете нерелевантные пары
    for sample in val_samples:
        if sample[2] == 0:
            phrase1 = sample[0]
            phrase2 = sample[1]
            if phrase1 in premise2samples:
                premise2samples[phrase1].append((phrase2, 0))

    # Добавим рандомных случайных сэмплов
    for sample in val_samples:
        phrase1 = sample[0]
        phrase2 = sample[1]
        # Добавляем вторую фразу как нерелевантный сэмпл к каждому левому предложению phrase1_1.
        for phrase1_1, samples_1 in premise2samples.items():
            if phrase1 != phrase1_1:
                if len(premise2samples[phrase1_1]) < 100:
                    if (phrase2, 0) not in premise2samples[phrase1_1] and (phrase2, 1) not in premise2samples[phrase1_1]:
                        premise2samples[phrase1_1].append((phrase2, 0))

    # Теперь в premise2samples для каждой фразы-ключа есть некоторое количество сравниваемых
    # фраз, из которых только 1 релевантна. Мы должны проверить, что модель именно эту пару
    # оценит максимально высоко, а остальным присвоит меньшую релевантность.
    nb_good = 0
    nb_total = 0

    pos_list = []
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

        # Ищем позицию сэмпла с релевантной парой
        true_pos = next(i for (i, z) in enumerate(phrase_y) if z[0][1] == 1)
        pos_list.append(true_pos)

    mean_pos = np.mean(pos_list)

    rank_accuracy = float(nb_good) / nb_total
    return rank_accuracy, mean_pos


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synonymy classifier trainer')
    parser.add_argument('--run_mode', choices='gridsearch train query eval'.split(), default='gridsearch')
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

    logging.info('Start using dataset_path={}'.format(dataset_path))

    # файл в формате json с найденными оптимальными параметрами классификатора,
    # создается в ходе gridsearch, используется для train
    best_params_path = os.path.join(tmp_dir, 'new_synonymy_detector.best_params.json')

    # файл с обученной моделью, создается в train, используется в query
    model_path = os.path.join(tmp_dir, 'new_synonymy_detector.model')

    df = pd.read_csv(dataset_path,
                     encoding='utf-8',
                     delimiter='\t',
                     index_col=None,
                     keep_default_na=False)

    samples = list(zip(df['premise'].values, df['question'].values, df['relevance'].values))

    if run_mode == 'gridsearch':
        logging.info('=== GRIDSEARCH ===')

        # На полном датасете подбор идет слишком долго, ограничим.
        samples = random.sample(samples, 10000)

        best_model_params = None
        best_score = 0.0
        crossval_count = 0

        for model_params in GridGenerator().generate():
            vectorizer = create_vectorizer(samples, model_params)

            logging.info(u'Cross-validation using model_params: {}'.format(get_params_str(model_params)))

            crossval_count += 1
            kf = KFold(n_splits=NFOLDS)
            scores = []
            mean_poses = []
            for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
                train_samples = [samples[i] for i in train_index]
                val_samples = [samples[i] for i in val_index]

                X_train, y_train = vectorize_data(train_samples, vectorizer, model_params)
                X_val, y_val = vectorize_data(val_samples, vectorizer, model_params)

                estimator = create_estimator(model_params)
                estimator.fit(X_train, y_train)
                score, mean_pos = compute_ranking_accuracy(estimator, vectorizer, model_params, val_samples)
                scores.append(score)
                mean_poses.append(mean_pos)

            score = np.mean(scores)
            score_std = np.std(scores)
            logging.info('Crossvalidation #{} precision@1={} std={} mean_pos={}'.format(crossval_count, score, score_std, np.mean(mean_poses)))

            if score > best_score:
                logging.info('!!! NEW BEST !!! precision@1={} params={}'.format(score, get_params_str(model_params)))
                best_score = score
                best_model_params = model_params.copy()
                with open(best_params_path, 'w') as f:
                    json.dump(best_model_params, f, indent=4)
            else:
                logging.info('No improvement over current best_score={}'.format(best_score))

        logging.info('best_score={} for model_params: {}'.format(best_score, get_params_str(best_model_params)))

    if run_mode == 'eval':
        # Оценка лучшей модели через кроссвалидацию.
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        vectorizer = create_vectorizer(samples, best_model_params)

        kf = KFold(n_splits=NFOLDS)
        scores = []
        mean_poses = []
        for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
            print('KFold[{}]'.format(ifold))
            train_samples = [samples[i] for i in train_index]
            val_samples = [samples[i] for i in val_index]

            X_train, y_train = vectorize_data(train_samples, vectorizer, best_model_params)
            X_val, y_val = vectorize_data(val_samples, vectorizer, best_model_params)

            estimator = create_estimator(best_model_params)
            estimator.fit(X_train, y_train)
            score, mean_pos = compute_ranking_accuracy(estimator, vectorizer, best_model_params, val_samples)
            scores.append(score)
            mean_poses.append(mean_pos)

        score = np.mean(scores)
        score_std = np.std(scores)
        logging.info('Crossvalidation precision@1={} std={} mean_pos={}'.format(score, score_std, np.mean(mean_poses)))

    if run_mode == 'train':
        logging.info('Loading best_model_params from "%s"', best_params_path)
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        best_vectorizer, X_data, y_data = data_vectorization(df, best_model_params)

        # финальное обучение классификатора на всех данных
        logging.info('Training the final classifier model_params: %s', get_params_str(best_model_params))
        estimator = create_estimator(best_model_params)
        estimator.fit(X_data, y_data)

        # сохраним натренированный классификатор и дополнительные параметры, необходимые для
        # использования модели в чатботе.
        model = {'vectorizer': best_vectorizer, 'estimator': estimator}
        logging.info('Storing model to "%s"', model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info('All done.')

    if run_mode == 'query':
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        logging.info('Restoring model from "{}"'.format(model_path))
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        while True:
            phrase1 = input('1:> ').strip()
            phrase2 = input('2:> ').strip()

            vectorizer = model['vectorizer']
            ps = vectorizer.transform([phrase1])
            qs = vectorizer.transform([phrase2])
            if best_model_params['random_proj'] == 0:
                nb_shingles = vectorizer.nb_shingles
                common_shingles = sets2matrix(and_setlists(ps, qs), nb_shingles)
                notmatched_ps = sets2matrix(sub_setlists(ps, qs), nb_shingles)
                notmatched_qs = sets2matrix(sub_setlists(qs, ps), nb_shingles)
                X_data = scipy.sparse.hstack([common_shingles, notmatched_ps, notmatched_qs])
            else:
                raise NotImplementedError()

            y_query = model['estimator'].predict(X_data)
            y = y_query[0]
            print(u'{}'.format(y))

    if run_mode == 'query2':
        data_folder = '../../data'
        tokenizer = rutokenizer.Tokenizer()
        tokenizer.load()

        # Проверка модели через поиск ближайших сэмплов в большом наборе предложений.
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        logging.info('Restoring model from "{}"'.format(model_path))
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        vectorizer = model['vectorizer']

        premises = []
        added_phrases = set()

        # поиск ближайшего приказа или вопроса из списка FAQ
        phrases2 = set()
        if True:
            for phrase in load_strings_from_yaml(os.path.join(data_folder, 'rules.yaml')):
                phrase2 = u' '.join(tokenizer.tokenize(phrase))
                if phrase2 not in added_phrases:
                    added_phrases.add(phrase2)
                    phrases2.add((phrase2, phrase))

        if True:
            with io.open(os.path.join(data_folder, 'intents.txt'), 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    phrase = line.strip()
                    if len(phrase) > 5 and not phrase.startswith('#') and u'_' not in phrase:
                        phrase2 = u' '.join(tokenizer.tokenize(phrase))
                        if phrase2 not in added_phrases:
                            added_phrases.add(phrase2)
                            phrases2.add((phrase2, phrase))

        if True:
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

        while True:
            phrase1 = input_kbd(':> ').strip()
            phrase1  = u' '.join(tokenizer.tokenize(phrase1))
            X_data = vectorize_data2([phrase1] * nb_phrases, [z[0] for z in phrases], vectorizer, best_model_params)
            y_query = model['estimator'].predict_proba(X_data)[:, 1]

            phrase_w = [(phrases[i][1], y_query[i]) for i in range(nb_phrases)]
            for phrase2, score in sorted(phrase_w, key=lambda z: -z[1])[:20]:
                print(u'{}\t{}'.format(phrase2, score))

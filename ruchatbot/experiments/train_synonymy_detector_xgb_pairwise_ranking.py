# coding: utf-8
"""
Эксперимент с альтернативной моделью детектора перефразировок на базе XGBRanker
(целевая функция rank_pairwise).
Представление сопоставляемых фраз - мешок шинглов или фрагментов от SentencePiece.

Общий сценарий использования:
1) gridsearch - находим параметры оптимальной модели, используя урезанный датасет
2) eval - оценка оптимальной модели через кроссвалидацию на полном датасете
3) train - тренировка лучшей модели и сохранение ее в pickle файле
4) query - ручная проверка натренированной модели

23-10-2019 первая реализация с XGBRanker
25-10-2019 добавлен вариант с LGBMRanker
"""

from __future__ import print_function
import pickle
import pandas as pd
import json
import os
import logging
import logging.handlers
import numpy as np
import argparse
import collections
import io
import yaml
import random
import tqdm

import scipy.sparse
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import xgboost as xgb
import lightgbm as lgb
import sentencepiece as spm

import rutokenizer


NFOLDS = 5
GROUP_SIZE = 30

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


def load_data(dataset_path, tmp_dir, max_groups):
    groups_filepath = os.path.join(tmp_dir, 'train_synonymy_detector_xgb_pairwise_ranking.groups_dataset.{}.pkl'.format(max_groups))
    if os.path.exists(groups_filepath):
        logging.info('Loading groups from "%s"', groups_filepath)
        with open(groups_filepath, 'rb') as f:
            groups = pickle.load(f)
            return groups
    else:
        logging.info('Loading data from "%s"...', dataset_path)
        df = pd.read_csv(dataset_path,
                         encoding='utf-8',
                         delimiter='\t',
                         index_col=None,
                         keep_default_na=False)

        premise2samples = dict()
        for i, r1 in df[df['relevance'] == 1].iterrows():
            phrase1 = r1['premise']
            phrase2 = r1['question']
            if phrase1 not in premise2samples:
                premise2samples[phrase1] = [(phrase2, 1)]
                # НАЧАЛО ОТЛАДКИ
                if len(premise2samples) >= max_groups:
                    break
                # КОНЕЦ ОТЛАДКИ

        logging.info('%d relevant pairs', len(premise2samples))

        # Добавим нерелевантные сэмплы.
        # Будем брать случайные вторые фразы из релевантных пар.
        df_1 = df[df['relevance'] == 1]

        for phrase1_1, samples in tqdm.tqdm(premise2samples.items(), desc='Add nonrelevant', total=len(premise2samples)):
            # Сейчас в samples есть только заданные в датасете позитивный и негативные
            # пары. Надо добавить негативных, чтобы размер группы стал заданным.
            samples_set = set(samples)

            additional_samples = set()
            for i, r2 in df_1.sample(GROUP_SIZE*10).iterrows():
                phrase1 = r2['premise']
                phrase2 = r2['question']
                if (phrase2, 0) not in samples_set and (phrase2, 1) not in samples_set:
                    additional_samples.add((phrase2, 0))

            additional_samples = list(additional_samples)
            additional_samples = random.sample(additional_samples, GROUP_SIZE - len(samples))
            samples.extend(additional_samples)

        # Теперь в premise2samples для каждой фразы-ключа есть некоторое количество сравниваемых
        # фраз, из которых только 1 релевантна. Мы должны проверить, что модель именно эту пару
        # оценит максимально высоко, а остальным присвоит меньшую релевантность.
        groups = []
        for phrase1, samples2 in premise2samples.items():
            group = []
            for phrase2, label in samples2:
                group.append((phrase1, phrase2, label))
            groups.append(group)

        logging.info('%d groups loaded', len(groups))

        logging.info('Saving groups as "%s"', groups_filepath)
        with open(groups_filepath, 'wb') as f:
            pickle.dump(groups, f)

        return groups


vectorizers = dict()


def enum_samples_in_groups(groups):
    for group in groups:
        for phrase1, phrase2, label in group:
            yield phrase1, phrase2, label


def create_vectorizer(groups, params):
    phrases = list(s[0] for s in enum_samples_in_groups(groups)) + list(s[1] for s in enum_samples_in_groups(groups))

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


def vectorize_data(groups, vectorizer, params):
    labels = [s[2] for s in enum_samples_in_groups(groups)]
    y_data = np.asarray(labels)

    phrases1 = [s[0] for s in enum_samples_in_groups(groups)]
    phrases2 = [s[1] for s in enum_samples_in_groups(groups)]
    group_data = list(map(len, groups))

    return vectorize_data2(phrases1, phrases2, vectorizer, params), y_data, group_data


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


def data_vectorization(groups, model_params):
    vectorizer = create_vectorizer(groups, model_params)
    X_data, y_data, group_data = vectorize_data(groups, vectorizer, model_params)
    return vectorizer, X_data, y_data, group_data


def create_estimator(model_params):
    if model_params['engine'] == 'XGBRanker':
        params = {'objective': 'rank:pairwise',
                  'learning_rate': model_params['learning_rate'],
                  #'gamma': 1.0,
                  #'min_child_weight': 0.1,
                  'max_depth': model_params['max_depth'],
                  'n_estimators': model_params['n_estimators']}
        model = xgb.sklearn.XGBRanker(**params)
        return model
    elif model_params['engine'] == 'LGBMRanker':
        params = {'objective': 'lambdarank',
                  'learning_rate': model_params['learning_rate'],
                  'max_depth': -1,
                  'n_estimators': model_params['n_estimators']}
        model = lgb.sklearn.LGBMRanker(**params)
        return model


class GridGenerator(object):
    def __init__(self):
        pass

    def vectorizer_grid(self):
        """ Подбор параметров подготовки данных и векторизации """

        for analyzer in ['char']:  # 'sentencepiece', 'char'
            if analyzer == 'sentencepiece':
                for vocab_size in [4000, 8000]:
                    for random_proj in [0]:
                        model_params['analyzer'] = analyzer
                        model_params['vocab_size'] = vocab_size
                        model_params['random_proj'] = random_proj
                        yield model_params
            else:
                for random_proj in [0]:
                    for min_shingle_len in [3]:
                        for max_shingle_len in [3]:
                            for min_df in [1]:
                                model_params = dict()
                                model_params['analyzer'] = analyzer
                                model_params['min_df'] = min_df
                                model_params['random_proj'] = random_proj
                                model_params['min_shingle_len'] = min_shingle_len
                                model_params['max_shingle_len'] = max_shingle_len
                                yield model_params

    def estimator_grid(self):
        """ Перебор параметров ранжировщика """
        for engine in ['XGBRanker']:  # 'LGBMRanker', 'XGBRanker'
            for learning_rate in [0.2, 0.25]:
                for n_estimators in [1500, 2000]:
                    for max_depth in [6, 7] if engine == 'XGBRanker' else [-1]:
                        model_params = dict()
                        model_params['engine'] = engine
                        model_params['n_estimators'] = n_estimators
                        model_params['learning_rate'] = learning_rate
                        model_params['max_depth'] = max_depth
                        yield model_params


def compute_ranking_accuracy(estimator, vectorizer, model_params, groups):
    nb_good = 0
    nb_total = 0
    pos_list = []

    for group in groups:
        X_data, y_data, _ = vectorize_data([group], vectorizer, model_params)

        y_pred = estimator.predict(X_data)
        maxy_pred = np.argmax(y_pred)
        maxy_data = np.argmax(y_data)
        nb_good += int(maxy_pred == maxy_data)
        nb_total += 1

        phrase_y = [(g, y) for g, y in zip(group, y_pred)]
        phrase_y = sorted(phrase_y, key=lambda z: -z[1])

        # Ищем позицию сэмпла с релевантной парой
        true_pos = next(i for (i, z) in enumerate(phrase_y) if z[0][2] == 1)
        pos_list.append(true_pos)

    precision1 = float(nb_good) / nb_total
    mrr = np.mean([1./(1.0+r) for r in pos_list])  # mean reciprocal rank
    return precision1, mrr


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
    parser.add_argument('--run_mode', choices='gridsearch eval train query'.split(), default='gridsearch')
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
    # создается в ходе gridsearch, используется для train и eval
    best_params_path = os.path.join(tmp_dir, 'train_synonymy_detector_xgb_pairwise_ranking.best_params.json')

    # файл с обученной моделью, создается в train, используется в query
    model_path = os.path.join(tmp_dir, 'train_synonymy_detector_xgb_pairwise_ranking.model')

    if run_mode == 'gridsearch':
        logging.info('=== GRIDSEARCH ===')

        groups = load_data(dataset_path, tmp_dir, 10000)

        cv_metric = 'precision@1'
        best_model_params = None
        best_score = 0.0
        crossval_count = 0

        for vectorizer_params in GridGenerator().vectorizer_grid():
            vectorizer = create_vectorizer(groups, vectorizer_params)
            for estimator_params in GridGenerator().estimator_grid():
                model_params = vectorizer_params.copy()
                model_params.update(estimator_params)
                logging.info(u'Cross-validation using model_params: {}'.format(get_params_str(model_params)))

                crossval_count += 1
                kf = KFold(n_splits=NFOLDS)
                scores = []
                mrrs = []
                for ifold, (train_index, val_index) in enumerate(kf.split(groups)):
                    # print('KFold[{}]'.format(ifold))
                    train_groups = [groups[i] for i in train_index]
                    val_groups = [groups[i] for i in val_index]

                    X_train, y_train, igroup_train = vectorize_data(train_groups, vectorizer, vectorizer_params)
                    X_val, y_val, igroup_val = vectorize_data(val_groups, vectorizer, vectorizer_params)

                    estimator = create_estimator(model_params)
                    estimator.fit(X=X_train, y=y_train, group=igroup_train)
                    precision1, mrr = compute_ranking_accuracy(estimator, vectorizer, model_params, val_groups)
                    scores.append(precision1)
                    mrrs.append(mrr)

                score = np.mean(scores)
                score_std = np.std(scores)
                logging.info(
                    'Crossvalidation #{} precision@1={} std={} mrr={}'.format(crossval_count, score, score_std,
                                                                                   np.mean(mrrs)))

                if score > best_score:
                    logging.info('!!! NEW BEST !!! {}={} params={}'.format(cv_metric, score, get_params_str(model_params)))
                    best_score = score
                    best_model_params = model_params.copy()
                    with open(best_params_path, 'w') as f:
                        json.dump(best_model_params, f, indent=4)
                else:
                    logging.info('No improvement over current best_{}={}'.format(cv_metric, best_score))

        logging.info('best_{}={} for model_params: {}'.format(cv_metric, best_score, get_params_str(best_model_params)))

    if run_mode == 'eval':
        # Оценка лучшей модели через кроссвалидацию.
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        groups = load_data(dataset_path, tmp_dir, 500000)

        vectorizer = create_vectorizer(groups, best_model_params)

        logging.info('Cross-validation started for params="%s"', get_params_str(best_model_params))
        kf = KFold(n_splits=NFOLDS)
        precisions = []
        mrrs = []
        for ifold, (train_index, val_index) in enumerate(kf.split(groups)):
            train_groups = [groups[i] for i in train_index]
            val_groups = [groups[i] for i in val_index]

            X_train, y_train, igroup_train = vectorize_data(train_groups, vectorizer, best_model_params)
            X_val, y_val, igroup_val = vectorize_data(val_groups, vectorizer, best_model_params)

            estimator = create_estimator(best_model_params)
            if best_model_params['engine'] == 'LGBMRanker':
                estimator.fit(X_train, y_train, group=igroup_train)
            elif best_model_params['engine'] == 'XGBRanker':
                estimator.fit(X_train, y_train, igroup_train)
            else:
                raise NotImplementedError()

            precision1, mrr = compute_ranking_accuracy(estimator, vectorizer, best_model_params, val_groups)
            logging.info('fold %d/%d precision@1=%f mrr=%f', ifold+1, NFOLDS, precision1, mrr)
            precisions.append(precision1)
            mrrs.append(mrr)

        precision = np.mean(precisions)
        score_std = np.std(precisions)
        logging.info('Cross-validation precision@1=%f std=%f mrr=%f', precision1, score_std, np.mean(mrr))

    if run_mode == 'train':
        groups = load_data(dataset_path, tmp_dir, 1000000)

        logging.info('Loading best_model_params from "%s"', best_params_path)
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        best_vectorizer, X_data, y_data, group_data = data_vectorization(groups, best_model_params)

        # финальное обучение классификатора на всех данных
        logging.info('Training the final classifier model_params: %s', get_params_str(best_model_params))
        estimator = create_estimator(best_model_params)
        estimator.fit(X_data, y_data, group_data)

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
            phrase1 = input(':> ').strip()
            phrase1  = u' '.join(tokenizer.tokenize(phrase1))
            X_data = vectorize_data2([phrase1] * nb_phrases, [z[0] for z in phrases], vectorizer, best_model_params)
            y_query = model['estimator'].predict_proba(X_data)[:, 1]

            phrase_w = [(phrases[i][1], y_query[i]) for i in range(nb_phrases)]
            for phrase2, score in sorted(phrase_w, key=lambda z: -z[1])[:20]:
                print(u'{}\t{}'.format(phrase2, score))

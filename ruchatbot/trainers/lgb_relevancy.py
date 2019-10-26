# -*- coding: utf-8 -*-
"""
Тренировка модели определения релевантности предпосылки и вопроса (--task relevancy)
и синонимичности (--task synonymy) на базе LightGBM.

Модель используется в проекте чат-бота https://github.com/Koziev/chatbot

Альтернативные модели - на базе XGBoost (xgb_relevancy.py) и нейросететевые (nn_relevancy.py,
nn_relevamcy_tripleloss.py)

Пример запуска обучения с нужными параметрами командной строки см. в ../scripts/train_lgb_relevancy.sh

В чатботе обученная данной программой модель используется классом LGB_RelevancyDetector
(https://github.com/Koziev/chatbot/blob/master/ruchatbot/bot/lgb_relevancy_detector.py)

30.12.2018 - добавлен эксперимент с SentencePiece моделью сегментации текста (https://github.com/google/sentencepiece)
01.01.2019 - добавлен эксперимент с StemPiece моделью сегментации текста
"""

from __future__ import division
from __future__ import print_function

import gc
import itertools
import json
import os
import io
import yaml
import random
import argparse
import codecs
import logging
import logging.handlers

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import lightgbm

import hyperopt
from hyperopt import hp, tpe, STATUS_OK, Trials

from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

#from trainers.evaluation_dataset import EvaluationDataset
#from trainers.evaluation_markup import EvaluationMarkup
from ruchatbot.utils.phrase_splitter import PhraseSplitter
import ruchatbot.utils.console_helpers
import ruchatbot.utils.logging_helpers


# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  # tpe.suggest OR hyperopt.rand.suggest

BEG_WORD = '\b'
END_WORD = '\n'


def ngrams(s, n):
    return [u''.join(z) for z in zip(*[s[i:] for i in range(n)])]


def str2shingles(s):
    return [u''.join(z) for z in zip(*[s[i:] for i in range(3)])]


def ngrams2(s, n):
    basic_shingles = str2shingles(s)
    words = s.split()
    for iword, word in enumerate(words):
        word_shingles = str2shingles(word)
        if iword > 0:
            prev_word = words[iword - 1]
            prev_trail = prev_word[0: 3]
            for word_shingle in word_shingles:
                new_shingle = u'~{}~+{}'.format(prev_trail, word_shingle)
                basic_shingles.append(new_shingle)

        if iword < len(words) - 1:
            next_word = words[iword + 1]
            next_trail = next_word[0: 3]
            for word_shingle in word_shingles:
                new_shingle = u'{}+~{}~'.format(word_shingle, next_trail)
                basic_shingles.append(new_shingle)

    return basic_shingles


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


if False:
    # эксперимент с SentencePiece моделью (https://github.com/google/sentencepiece)
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    rc = sp.Load("/home/inkoziev/github/sentencepiece/ru_raw.model")
    print('rc={}'.format(rc))

    from nltk.stem.snowball import RussianStemmer, EnglishStemmer

    # Еще эксперимент - использование стеммера для получения "StemPiece" сегментации текста.
    stemmer = RussianStemmer()

    def word2pieces(word):
        if len(word) < 3:
            return [word]

        stem = stemmer.stem(word)
        assert(len(stem) > 0)
        ending = word[len(stem):]
        if len(ending) > 0:
            return [stem, '~' + ending]
        else:
            return [stem]


def str2shingles(s):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(3)])]


def ngrams3(phrase, n):
    #return list(itertools.chain(stemmer.stem(word) for word in phrase.split(' '))) + str2shingles(phrase)
    #return str2shingles(phrase)
    return sp.EncodeAsPieces(phrase)


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
            X_data[idata, icol + shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_ps:
        if shingle in shingle2id:
            X_data[idata, icol + shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_qs:
        if shingle in shingle2id:
            X_data[idata, icol + shingle2id[shingle]] = True


def train_model(lgb_params, D_train, D_val, y_val):
    """
    Тренировка модели на данных D_train, валидация и early stopping на D_val и y_val.
    :param lgb_params: параметры тренировки для LightGBM
    :param D_train: тренировочные входные данные
    :param D_val: данные для валидации
    :param y_val: целевые значения для валидационного набора для расчета accuracy и F1
    :return: кортеж (бустер, acc, f1)
    """
    lgb_params['bagging_freq'] = 1

    logging.info('Train LightGBM model with learning_rate={} num_leaves={} min_data_in_leaf={} bagging_fraction={}...'.format(lgb_params['learning_rate'],
                                                                                                                              lgb_params['num_leaves'],
                                                                                                                              lgb_params['min_data_in_leaf'],
                                                                                                                              lgb_params['bagging_fraction']))
    cl = lightgbm.train(lgb_params,
                        D_train,
                        valid_sets=[D_val],
                        valid_names=['val'],
                        num_boost_round=5000,
                        verbose_eval=50,
                        early_stopping_rounds=50)

    y_pred = cl.predict(X_val)
    y_pred = (y_pred >= 0.5).astype(np.int)

    # Точность на валидационных данных малоинформативна из-за сильного дисбаланса 1/0 классов,
    # напечатаем только для контроля кода обучения.
    acc = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred)

    # из-за сильного дисбаланса (в пользу исходов с y=0) оценивать качество
    # получающейся модели лучше по f1
    f1 = sklearn.metrics.f1_score(y_true=y_val, y_pred=y_pred)

    return cl, acc, f1


def evaluate_model(lgb_relevancy, model_config, eval_data, verbose):
    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']

    nb_good = 0  # попадание предпосылки в top-1
    nb_good5 = 0
    nb_good10 = 0
    nb_total = 0

    wrt = None
    if verbose:
        wrt = codecs.open(os.path.join(tmp_folder, 'lgb_relevancy.evaluation.txt'), 'w', 'utf-8')

    for irecord, phrases in eval_data.generate_groups():
        nb_samples = len(phrases)

        X_data = lil_matrix((nb_samples, xgb_relevancy_nb_features), dtype='float32')

        for irow, (premise_words, question_words) in enumerate(phrases):
            premise_wx = premise_words
            question_wx = question_words

            premise_shingles = set(ngrams(premise_wx, xgb_relevancy_shingle_len))
            question_shingles = set(ngrams(question_wx, xgb_relevancy_shingle_len))

            vectorize_sample_x(X_data, irow, premise_shingles, question_shingles, xgb_relevancy_shingle2id)

        y_pred = lgb_relevancy.predict(X_data)

        # предпосылка с максимальной релевантностью
        max_index = np.argmax(y_pred)
        selected_premise = phrases[max_index][0]

        nb_total += 1
        # эта выбранная предпосылка соответствует одному из вариантов
        # релевантных предпосылок в этой группе?
        if eval_data.is_relevant_premise(irecord, selected_premise):
            nb_good += 1
            nb_good5 += 1
            nb_good10 += 1
            if verbose:
                print(EvaluationMarkup.ok_color + EvaluationMarkup.ok_bullet + EvaluationMarkup.close_color, end='')
                wrt.write(EvaluationMarkup.ok_bullet)
        else:
            if verbose:
                print(EvaluationMarkup.fail_color + EvaluationMarkup.fail_bullet + EvaluationMarkup.close_color, end='')
                wrt.write(EvaluationMarkup.fail_bullet)
            # среди top-5 или top-10 предпосылок есть верная?
            sorted_phrases = [x for x, _ in sorted(itertools.izip(phrases, y_pred), key=lambda z:-z[1])]

            for i in range(1, 10):
                selected_premise = sorted_phrases[i][0]
                if eval_data.is_relevant_premise(irecord, selected_premise):
                    if i < 5:
                        nb_good5 += 1  # верная предпосылка вошла в top-5
                    if i < 10:
                        nb_good10 += 1
                    break

        if verbose == 1:
            message_line = u'{:<40} {:<40} {}/{}'.format(question_words, phrases[max_index][0], y_pred[max_index], y_pred[0])
            print(message_line)
            wrt.write(message_line + u'\n')

        # для отладки: top релевантных вопросов
        if False:
            print(u'Most similar premises for question {}'.format(question))
            yy = [(y_pred[i], i) for i in range(len(y_pred))]
            yy = sorted(yy, key=lambda z: -z[0])

            for sim, index in yy[:5]:
                print(u'{:.4f} {}'.format(sim, phrases[index][0]))

    if wrt is not None:
        wrt.close()

    # Итоговая точность выбора предпосылки.
    accuracy = float(nb_good) / float(nb_total)
    if verbose == 1:
        print('accuracy       ={}'.format(accuracy))

        # Также выведем точность попадания верной предпосылки в top-5 и top-10
        accuracy5 = float(nb_good5) / float(nb_total)
        print('accuracy top-5 ={}'.format(accuracy5))

        accuracy10 = float(nb_good10) / float(nb_total)
        print('accuracy top-10={}'.format(accuracy10))

    return accuracy


def calc_ranking_measures(df, estimator, nb_features, shingle_len, shingle2id):
    # Код для получения оценочной метрики "качество ранжирования".
    # Берем условное val-подмножество из базового датасета.
    df_val = df.sample(n=10000, random_state=12345678)
    premise2samples = dict()
    for i, r1 in df_val[df_val['relevance'] == 1].iterrows():
        phrase1 = r1['premise']
        phrase2 = r1['question']
        if phrase1 not in premise2samples:
            premise2samples[phrase1] = [(phrase2, 1)]

    # 23-10-2019 добавим готовые негативные примеры из датасета.
    for i, r1 in df_val[df_val['relevance'] == 0].iterrows():
        phrase1 = r1['premise']
        phrase2 = r1['question']
        if phrase1 in premise2samples:
            premise2samples[phrase1].append((phrase2, 1))

    for i, r2 in df_val[df_val['relevance'] == 0].iterrows():
        phrase1 = r2['premise']
        phrase2 = r2['question']
        # Добавляем вторую фразу как нерелевантный сэмпл к каждому левому предложению.
        for phrase1_1, samples_1 in premise2samples.items():
            if phrase1 != phrase1_1:
                if len(premise2samples[phrase1_1]) < 100:
                    phrases2 = premise2samples[phrase1_1]
                    if (phrase2, 0) not in phrases2 and (phrase2, 1) not in phrases2:
                        premise2samples[phrase1_1].append((phrase2, 0))

    # Теперь в premise2samples для каждой фразы-ключа есть некоторое количество сравниваемых
    # фраз, из которых только 1 релевантна. Мы должны проверить, что модель именно эту пару
    # оценит максимально высоко, а остальным присвоит меньшую релевантность.
    # 07-07-2019 Кроме того, можно брать позицию правильного выбора после сортировки по релевантности.
    # Чем ближе средняя позиция к 0, тем лучше модель
    nb_good = 0
    nb_total = 0
    rank_positions = []  # тут накопим позиции правильного сэмпла при ранжировке
    for phrase1, samples in premise2samples.items():
        df_samples = pd.DataFrame(index=None, columns=['premise', 'question', 'relevance'])
        for phrase2, label in samples:
            df_samples = df_samples.append({'premise': phrase1, 'question': phrase2, 'relevance': label},
                                           ignore_index=True)

        # X_data, y_data = vectorize_data(df_samples, best_vectorizer, best_model_params)

        nb_samples = len(samples)
        X_data = lil_matrix((nb_samples, nb_features), dtype='float32')
        y_data = df_samples['relevance'].values

        for irow, (premise, question) in enumerate(
                zip(df_samples['premise'].values, df_samples['question'].values)):
            premise_shingles = set(ngrams(premise, shingle_len))
            question_shingles = set(ngrams(question, shingle_len))
            vectorize_sample_x(X_data, irow, premise_shingles, question_shingles, shingle2id)

        y_pred = estimator.predict(X_data)
        maxy_pred = np.argmax(y_pred)
        maxy_data = np.argmax(y_data)
        nb_good += int(maxy_pred == maxy_data)
        nb_total += 1

        yy = [(y_pred[i], y_data[i]) for i in range(nb_samples)]
        yy = sorted(yy, key=lambda z: -z[0])
        y_true_pos = next(i for i, z in enumerate(yy) if z[1] == 1)
        rank_positions.append(y_true_pos)


    # Precision@1 - для какой доли сэмплов правильная пара попадает в top-1
    rank_accuracy = float(nb_good) / nb_total

    # Средняя позиция правильного ответа
    mean_pos = np.mean(rank_positions)

    return rank_accuracy, mean_pos


def get_params(space):
    px = dict()
    px['boosting_type'] = 'gbdt'
    px['objective'] = 'binary'
    px['metric'] = 'binary_logloss'
    px['learning_rate'] = space['learning_rate']
    px['num_leaves'] = int(space['num_leaves'])
    px['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    px['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    px['max_depth'] = int(space['max_depth']) if 'max_depth' in space else -1
    px['lambda_l1'] = 0.0  # space['lambda_l1'],
    px['lambda_l2'] = 0.0  # space['lambda_l2'],
    px['max_bin'] = 256
    px['feature_fraction'] = space['feature_fraction']
    px['bagging_fraction'] = space['bagging_fraction']
    px['bagging_freq'] = 1

    return px


obj_call_count = 0
cur_best_acc = -np.inf
hyperopt_log_writer = None
ho_model_config = None
ho_eval_data = None


def objective(space):
    global obj_call_count, cur_best_acc

    obj_call_count += 1

    logging.info('\nLightGBM objective call #{} cur_best_acc={:7.5f}'.format(obj_call_count, cur_best_acc))

    lgb_params = get_params(space)

    sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
    logging.info('Params: {}'.format(str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])))

    cl, val_acc, val_f1 = train_model(lgb_params, D_train, D_val, y_val)
    eval_acc = evaluate_model(cl, ho_model_config, ho_eval_data, 0)
    logging.info('eval_acc={}'.format(eval_acc))

    do_store = False
    if eval_acc > cur_best_acc:
        cur_best_acc = eval_acc
        do_store = True
        print(EvaluationMarkup.ok_color + 'NEW BEST ACC={}'.format(cur_best_acc) + EvaluationMarkup.close_color)

    prefix = '   '
    if do_store:
        model_filename = ho_model_config['model_filename']
        cl.save_model(model_filename)
        prefix = '(*)'

    hyperopt_log_writer.write('{}eval acc={:<7.5f} Params:{}\n'.format(prefix,
                                                                       eval_acc,
                                                                       str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])))
    hyperopt_log_writer.flush()

    return{'loss': -cur_best_acc, 'status': STATUS_OK}


# -------------------------------------------------------------------


parser = argparse.ArgumentParser(description='LightGBM classifier for text relevance estimation')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | evaluate | query | query2 | hardnegative')
parser.add_argument('--hyperopt', type=int, default=0, help='use hyperopt when training')
parser.add_argument('--shingle_len', type=int, default=3, help='shingle length')
parser.add_argument('--eta', type=float, default=0.184, help='"eta" (learning rate) parameter for LightGBM')
parser.add_argument('--subsample', type=float, default=0.997, help='"subsample" parameter for LightGBM')
parser.add_argument('--num_leaves', type=int, default=48, help='"num_leaves" parameter for LightGBM')
parser.add_argument('--min_data_in_leaf', type=int, default=73, help='"min_data_in_leaf" parameter for LightGBM')
parser.add_argument('--input', type=str, default='../data/premise_question_relevancy.csv', help='path to input dataset')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')
parser.add_argument('--lemmatize', type=int, default=0, help='canonize phrases before extracting the shingles: 0 - none, 1 - lemmas, 2 - stems')
parser.add_argument('--task', type=str, default='relevancy', choices='relevancy synonymy'.split(), help='model filenames keyword')

args = parser.parse_args()

input_path = args.input
tmp_folder = args.tmp
data_folder = args.data_dir
run_mode = args.run_mode
lemmatize = args.lemmatize
subsample = args.subsample
num_leaves = args.num_leaves
min_data_in_leaf = args.min_data_in_leaf
task = args.task

config_filename = 'lgb_{}.config'.format(task)

# количество случайных наборов параметров, проверяемых в hyperopt
# если указать 0, то hyperopt не применяется, а выполняется обучение
# с заданными параметрами (--num_leaves, --min_data_in_leaf, --eta, --subsample)
use_hyperopt = args.hyperopt

# основной настроечный параметр модели - длина символьных N-грамм (шинглов)
shingle_len = args.shingle_len
if shingle_len < 2 or shingle_len > 6:
    print('Invalid --shingle_len option value')
    exit(1)

eta = args.eta
if eta < 0.01 or eta >= 1.0:
    print('Invalid --eta option value')
    exit(1)


# настраиваем логирование в файл
ruchatbot.utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'lgb_{}.log'.format(task)))


if run_mode == 'train':
    # Режим тренировки модели.
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    logging.info('Input dataset loaded, samples.count={}'.format(df.shape[0]))

    tokenizer = PhraseSplitter.create_splitter(lemmatize)

    all_shingles = set()

    for i, record in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Shingles'):
        for phrase in [record['premise'], record['question']]:
            #if phrase.startswith(u'как меня зовут'):
            #    pass

            words = tokenizer.tokenize(phrase)
            wx = words2str(words)
            all_shingles.update(ngrams(wx, shingle_len))

    nb_shingles = len(all_shingles)
    logging.info('nb_shingles={}'.format(nb_shingles))

    shingle2id = dict([(s, i) for i, s in enumerate(all_shingles)])

    phrases = []
    ys = []
    weights = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
        weights.append(row['weight'])
        phrase1 = row['premise']
        phrase2 = row['question']
        words1 = words2str(tokenizer.tokenize(phrase1))
        words2 = words2str(tokenizer.tokenize(phrase2))

        y = row['relevance']
        if y in (0, 1):
            ys.append(y)
            phrases.append((words1, words2, phrase1, phrase2))

    nb_patterns = len(ys)

    nb_features = nb_shingles * 3
    X_data = lil_matrix((nb_patterns, nb_features), dtype='float32')
    y_data = []

    for idata, (phrase12, y12) in tqdm.tqdm(enumerate(itertools.izip(phrases, ys)),
                                            total=nb_patterns,
                                            desc='Vectorization'):
        premise = phrase12[0]
        question = phrase12[1]
        y = y12

        y_data.append(y)

        premise_shingles = ngrams(premise, shingle_len)
        question_shingles = ngrams(question, shingle_len)
        vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, shingle2id)

    nb_0 = len(filter(lambda y: y == 0, y_data))
    nb_1 = len(filter(lambda y: y == 1, y_data))

    logging.info('nb_0={}'.format(nb_0))
    logging.info('nb_1={}'.format(nb_1))

    SEED = 123456
    TEST_SHARE = 0.2
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_data,
                                                                      y_data,
                                                                      weights,
                                                                      test_size=TEST_SHARE,
                                                                      random_state=SEED)

    D_train = lightgbm.Dataset(data=X_train, label=y_train, weight=w_train, silent=1)
    D_val = lightgbm.Dataset(data=X_val, label=y_val, weight=w_val, silent=1)

    gc.collect()

    model_filename = os.path.join(tmp_folder, 'lgb_{}.model'.format(task))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {'model': 'lightgbm',
                    'shingle2id': shingle2id,
                    'model_filename': model_filename,
                    'shingle_len': shingle_len,
                    'nb_features': nb_features,
                    'lemmatize': lemmatize
                    }

    with open(os.path.join(tmp_folder, config_filename), 'w') as f:
        json.dump(model_config, f, indent=4)

    if use_hyperopt:
        ho_model_config = model_config

        ho_eval_data = EvaluationDataset(0, tokenizer, 'none')
        ho_eval_data.load(data_folder)

        space = {'num_leaves': hp.quniform('num_leaves', 20, 100, 1),
                 'min_data_in_leaf': hp.quniform('min_data_in_leaf', 5, 100, 1),
                 'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
                 'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
                 'learning_rate': hp.loguniform('learning_rate', -2, -1.2),
                 'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
                 }

        hyperopt_log_writer = open(os.path.join(tmp_folder, 'lgb_{}.hyperopt.txt'.format(task)), 'w')

        trials = Trials()
        best = hyperopt.fmin(fn=objective,
                             space=space,
                             algo=HYPEROPT_ALGO,
                             max_evals=500,
                             trials=trials,
                             verbose=1)

        hyperopt_log_writer.close()
    else:
        lgb_params = dict()
        lgb_params['boosting_type'] = 'gbdt'
        lgb_params['objective'] = 'binary'
        lgb_params['metric'] = 'binary_logloss'
        lgb_params['learning_rate'] = eta
        lgb_params['num_leaves'] = num_leaves
        lgb_params['min_data_in_leaf'] = min_data_in_leaf
        lgb_params['min_sum_hessian_in_leaf'] = 1
        lgb_params['max_depth'] = -1
        lgb_params['lambda_l1'] = 0.0  # space['lambda_l1'],
        lgb_params['lambda_l2'] = 0.0  # space['lambda_l2'],
        lgb_params['max_bin'] = 256
        lgb_params['feature_fraction'] = 0.950673776143  # 1.0
        lgb_params['bagging_fraction'] = subsample
        lgb_params['bagging_freq'] = 1

        cl, acc, f1 = train_model(lgb_params, D_train, D_val, y_val)

        logging.info('Training has finished')
        logging.info('val acc={}'.format(acc))
        logging.info('val f1={}'.format(f1))

        # сохраняем саму модель
        cl.save_model(model_filename)

        estimator = cl
        #tokenizer = PhraseSplitter.create_splitter(model_config['lemmatize'])
        #nb_features = model_config['nb_features']
        #shingle_len = model_config['shingle_len']
        #shingle2id = model_config['shingle2id']

        # eval_data = EvaluationDataset(0, tokenizer, 'none')
        # eval_data.load(data_folder)
        #
        # accuracy = evaluate_model(lgb_relevancy, model_config, eval_data, verbose=1)
        #
        # # Итоговая точность выбора предпосылок на оценочной задаче.
        # print('eval accuracy={}'.format(accuracy))

        logging.info('Calculate ranking accuracy...')
        rank_acc, mean_pos = calc_ranking_measures(df, estimator, nb_features, shingle_len, shingle2id)
        logging.info('Ranking accuracy: precision@1={} mean_pos={}'.format(rank_acc, mean_pos))

        # Для отладки - прогоним через модель весь датасет и сохраним результаты в текстовый файл.
        # y_pred = cl.predict(X_data)
        # with codecs.open(os.path.join(tmp_folder, 'lgb_{}.validation.txt'.format(task)), 'w', 'utf-8') as wrt:
        #     for i in range(len(y_pred)):
        #         premise = phrases[i][2]
        #         question = phrases[i][3]
        #         wrt.write(u'{}\n{}\ny_true={} y_pred={}\n\n'.format(premise, question, y_data[i], y_pred[i]))

if run_mode == 'query':
    # Ручная проверка модели на вводимых в консоли предпосылках и вопросах.

    # Загружаем данные обученной модели.
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    tokenizer = PhraseSplitter.create_splitter(model_config['lemmatize'])

    lgb_relevancy = lightgbm.Booster(model_file=model_config['model_filename'])

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmalize = model_config['lemmatize']

    while True:
        X_data = lil_matrix((1, xgb_relevancy_nb_features), dtype='float32')

        premise = ruchatbot.utils.console_helpers.input_kbd('premise:> ').strip().lower()
        if len(premise) == 0:
            break
        question = ruchatbot.utils.console_helpers.input_kbd('question:> ').strip().lower()

        premise_wx = words2str(tokenizer.tokenize(premise))
        question_wx = words2str(tokenizer.tokenize(question))

        premise_shingles = set(ngrams(premise_wx, xgb_relevancy_shingle_len))
        question_shingles = set(ngrams(question_wx, xgb_relevancy_shingle_len))

        vectorize_sample_x(X_data, 0, premise_shingles, question_shingles, xgb_relevancy_shingle2id)

        y_pred = lgb_relevancy.predict(X_data)
        print('{}\n\n'.format(y_pred[0]))

if run_mode == 'query2':
    # Ручная проверка модели на вводимых в консоли вопросах.
    # Список предпосылок читается из заданного файла.

    # Загружаем данные обученной модели.
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    tokenizer = PhraseSplitter.create_splitter(model_config['lemmatize'])

    lgb_relevancy = lightgbm.Booster(model_file=model_config['model_filename'])

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmalize = model_config['lemmatize']

    premises = []

    added_phrases = set()
    if task == 'relevancy':
        # Поиск лучшей предпосылки, релевантной введенному вопросу

        if True:
            for fname in ['profile_facts_1.dat']:
                with codecs.open(os.path.join(data_folder, fname), 'r', 'utf-8') as rdr:
                    for line in rdr:
                        phrase = line.strip()
                        if phrase.startswith('#'):
                            continue

                        if len(phrase) > 5:
                            phrase2 = u' '.join(tokenizer.tokenize(phrase))
                            if phrase2 not in added_phrases:
                                added_phrases.add(phrase2)
                                premises.append((phrase2, phrase))

        if True:
            # Для hard negative mining берем все предпосылки из датасета PQA
            df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
            all_premises = df['premise'].unique()
            print('{} premises loaded from {}'.format(len(all_premises), input_path))
            premises.extend((premise, premise) for premise in all_premises)

        if True:
            for phrase in load_strings_from_yaml(os.path.join(data_folder, 'rules.yaml')):
                phrase2 = u' '.join(tokenizer.tokenize(phrase))
                if phrase2 not in added_phrases:
                    added_phrases.add(phrase2)
                    premises.append((phrase2, phrase))


    elif task == 'synonymy':
        # поиск ближайшего приказа или вопроса из списка FAQ
        phrases2 = set()
        if True:
            for phrase in load_strings_from_yaml(os.path.join(data_folder, 'rules.yaml')):
                phrase2 = u' '.join(tokenizer.tokenize(phrase))
                if phrase2 not in added_phrases:
                    added_phrases.add(phrase2)
                    phrases2.add((phrase2, phrase))

        if True:
            with codecs.open(os.path.join(data_folder, 'intents.txt'), 'r', 'utf-8') as rdr:
                for line in rdr:
                    phrase = line.strip()
                    if len(phrase) > 5 and not phrase.startswith('#') and u'_' not in phrase:
                        phrase2 = u' '.join(tokenizer.tokenize(phrase))
                        if phrase2 not in added_phrases:
                            added_phrases.add(phrase2)
                            phrases2.add((phrase2, phrase))

        if True:
            with codecs.open(os.path.join(data_folder, 'faq2.txt'), 'r', 'utf-8') as rdr:
                for line in rdr:
                    phrase = line.strip()
                    if len(phrase) > 5 and phrase.startswith(u'Q:'):
                        phrase = phrase.replace(u'Q:', u'').strip()
                        phrase2 = u' '.join(tokenizer.tokenize(phrase))
                        if phrase2 not in added_phrases:
                            added_phrases.add(phrase2)
                            phrases2.add((phrase2, phrase))

        premises = list(phrases2)
    else:
        raise NotImplementedError()

    nb_premises = len(premises)
    print('nb_premises={}'.format(nb_premises))

    while True:
        X_data = lil_matrix((nb_premises, xgb_relevancy_nb_features), dtype='float32')

        question = ruchatbot.utils.console_helpers.input_kbd('question:> ').strip().lower()
        if len(question) == 0:
            break

        question_wx = words2str(tokenizer.tokenize(question))
        question_shingles = set(ngrams(question_wx, xgb_relevancy_shingle_len))

        for ipremise, premise in enumerate(premises):
            premise_wx = words2str(tokenizer.tokenize(premise[0]))
            premise_shingles = set(ngrams(premise_wx, xgb_relevancy_shingle_len))
            vectorize_sample_x(X_data, ipremise, premise_shingles, question_shingles, xgb_relevancy_shingle2id)

        y_pred = lgb_relevancy.predict(X_data)
        phrase_rels = [(premises[i][1], y_pred[i]) for i in range(nb_premises)]
        phrase_rels = sorted(phrase_rels, key=lambda z: -z[1])
        print('-'*50)
        for phrase, sim in phrase_rels[:30]:  # выводим топ ближайших фраз
            print(u'{:6.4f} {}'.format(sim, phrase))

if run_mode == 'evaluate':
    # Оценка качества натренированной модели на специальном наборе вопросов и
    # ожидаемых выборов предпосылок из отдельного тренировочного набора.

    # Загружаем данные обученной модели.
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    estimator = lightgbm.Booster(model_file=model_config['model_filename'])
    tokenizer = PhraseSplitter.create_splitter(model_config['lemmatize'])
    nb_features = model_config['nb_features']
    shingle_len = model_config['shingle_len']
    shingle2id = model_config['shingle2id']

    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

    logging.info('Calculate ranking accuracy...')
    rank_acc, mean_pos = calc_ranking_measures(df, estimator, nb_features, shingle_len, shingle2id)
    logging.info('Ranking accuracy: precision@1={} mean_pos={}'.format(rank_acc, mean_pos))

if run_mode == 'clusterize':
    # семантическая кластеризация предложений с использованием
    # обученной модели в качестве калькулятора метрики попарной близости.

    # Загружаем данные обученной модели.
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    lgb_relevancy_shingle2id = model_config['shingle2id']
    lgb_relevancy_shingle_len = model_config['shingle_len']
    lgb_relevancy_nb_features = model_config['nb_features']
    lgb_relevancy_lemmalize = model_config['lemmatize']

    tokenizer = PhraseSplitter.create_splitter(lgb_relevancy_lemmalize)

    lgb_relevancy = lightgbm.Booster(model_file=model_config['model_filename'])

    # в качестве источника предложений возьмем обучающий датасет. из которого возьмем
    # релевантные предпосылки и вопросы
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)

    phrases = set()
    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
        if row['relevance'] == 1:
            for phrase in [row['question'], row['premise']]:
                words = tokenizer.tokenize(phrase)
                wx = words2str(words)
                phrases.add((wx, phrase))

    # оставим небольшую часть предложений, чтобы ограничить количество попарных дистанций
    phrases = np.random.permutation(list(phrases))[:2000]
    nb_phrases = len(phrases)

    print('Computation of {0}*{0} distance matrix'.format(nb_phrases))

    distances = np.zeros((nb_phrases, nb_phrases), dtype='float32')

    min_dist = np.inf
    max_dist = -np.inf

    # в принципе, достаточно вычислить верхнетреугольную матрицу расстояний.
    for i1, (phrase1, _) in tqdm.tqdm(enumerate(phrases[:-1]), total=nb_phrases - 1, desc='Distance matrix'):
        shingles1 = set(ngrams(phrase1, lgb_relevancy_shingle_len))
        n2 = nb_phrases - i1 - 1
        X_data = lil_matrix((n2, lgb_relevancy_nb_features), dtype='float32')

        for i2, (phrase2, _) in enumerate(phrases[i1 + 1:]):
            shingles2 = set(ngrams(phrase2, lgb_relevancy_shingle_len))
            vectorize_sample_x(X_data, i2, shingles1, shingles2, lgb_relevancy_shingle2id)

        y_pred = lgb_relevancy.predict(X_data)
        for i2 in range(i1 + 1, nb_phrases):
            y = 1.0 - y_pred[i2 - i1 - 1]
            distances[i1, i2] = y
            distances[i2, i1] = y
            min_dist = min(min_dist, y)
            max_dist = max(max_dist, y)

    print('\nmin_dist={} max_dist={}'.format(min_dist, max_dist))

    print('Clusterization...')
    if False:
        # http://scikit-learn.org/dev/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
        cl = sklearn.cluster.DBSCAN(eps=0.1, min_samples=5, metric='precomputed',
                                    metric_params=None, algorithm='auto',
                                    leaf_size=10, p=None, n_jobs=2)

        db = cl.fit(distances)
        labels = db.labels_
    else:
        cl = sklearn.cluster.AgglomerativeClustering(n_clusters=400, affinity='precomputed',
                                                     memory=None, connectivity=None,
                                                     compute_full_tree='auto', linkage='complete')
        cl.fit(distances)
        labels = cl.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Number of clusters={}'.format(n_clusters_))

    with codecs.open(os.path.join(tmp_folder, 'lgb_relevancy_clusters.txt'), 'w', 'utf-8') as wrt:
        for icluster in range(n_clusters_):
            wrt.write('=== CLUSTER #{} ===\n'.format(icluster))

            for iphrase, label in enumerate(labels):
                if label == icluster:
                    wrt.write(u'{}\n'.format(phrases[iphrase][1]))

            wrt.write('\n\n')

if run_mode == 'hardnegative':
    # Поиск новых негативных сэмплов, которые надо добавить в датасет для
    # уменьшения количества неверно определяемых положительных пар.
    # Алгоритм: для сэмпла из ручного датасета определяем релевантность к остальным
    # репликам в этом же датасете. Отбираем те реплики, которые дают оценку
    # релевантности >0.5, исключаем правильные положительные и известные негативные,
    # остаются те фразы, которые считаются релевантными исходной фразе, но это неверно.
    # Сохраняем получающийся список в файле для ручной модерации.

    # Загружаем данные обученной модели.
    with open(os.path.join(tmp_folder, config_filename), 'r') as f:
        model_config = json.load(f)

    tokenizer = PhraseSplitter.create_splitter(model_config['lemmatize'])

    lgb_relevancy = lightgbm.Booster(model_file=model_config['model_filename'])

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']
    xgb_relevancy_lemmalize = model_config['lemmatize']

    known_pairs = set()
    test_phrases = set()
    if task == 'synonymy':
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
    print('nb_premises={}'.format(nb_premises))

    with io.open(os.path.join(tmp_folder, 'lgb_relevancy.hard_negatives.txt'), 'w', encoding='utf-8') as wrt:
        nb_stored = 0  # кол-во найденных и сохраненных негативных примеров
        for iphrase, (nphrase, question) in enumerate(sorted(premises, key=lambda _: random.random())):
            X_data = lil_matrix((nb_premises, xgb_relevancy_nb_features), dtype='float32')
            question_wx = words2str(tokenizer.tokenize(question))
            question_shingles = set(ngrams(question_wx, xgb_relevancy_shingle_len))

            for ipremise, premise in enumerate(premises):
                premise_wx = words2str(tokenizer.tokenize(premise[0]))
                premise_shingles = set(ngrams(premise_wx, xgb_relevancy_shingle_len))
                vectorize_sample_x(X_data, ipremise, premise_shingles, question_shingles, xgb_relevancy_shingle2id)

            selected_phrases2 = []
            y_pred = lgb_relevancy.predict(X_data)
            phrase_rels = [(premises[i][1], y_pred[i]) for i in range(nb_premises) if y_pred[i] > 0.5]
            phrase_rels = sorted(phrase_rels, key=lambda z: -z[1])
            for phrase2, sim in phrase_rels[:20]:
                if phrase2 != question and (question, phrase2) not in known_pairs and (phrase2, question) not in known_pairs:
                    selected_phrases2.append(phrase2)
            if len(selected_phrases2) > 0:
                wrt.write(u'{}\n'.format(question))
                for phrase2 in selected_phrases2:
                    wrt.write(u'(-) {}\n'.format(phrase2))
                wrt.write(u'\n\n')
                wrt.flush()
                nb_stored += len(selected_phrases2)
                print('{}/{} processed, {} negative samples stored'.format(iphrase, nb_premises, nb_stored))

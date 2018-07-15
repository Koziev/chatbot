# -*- coding: utf-8 -*-
'''
Тренировка модели, которая посимвольно генерирует ответ для заданной предпосылки и вопроса.
Используется XGB.
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import itertools
import json
import os
import sys
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection

from utils.tokenizer import Tokenizer

SHINGLE_LEN = 3

PHRASE_DIM = 0  #1024  # длина вектора SDR предложения; 0 если SDR не использовать

NB_PREV_CHARS = 5 # кол-во пред. символов в сгенерированном ответе, учитываемых при выборе следующего символа.

NB_SAMPLES = 1000000 # кол-во записей в датасете (до разбивки на тренировку и валидацию)

MIN_SHINGLE_FREQ = 5

NB_TREES = 1000
MAX_DEPTH = 5

BEG_LEN = 15  # длина в символах начального фрагмента фраз, который дает отдельные фичи
END_LEN = 15  # длина с символах конечного фрагмента фраз


# -------------------------------------------------------------------

input_path = '../data/premise_question_answer.csv'
tmp_folder = '../tmp'
data_folder = '../data'

# -------------------------------------------------------------------

BEG_WORD = u'\b'
END_WORD = u'\n'

BEG_CHAR = u'\b'
END_CHAR = u'\n'

# -------------------------------------------------------------------


def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    """
    Цепочку слов соединяем в строку, добавляя перед цепочкой и после нее
    пробел и специальные символы начала и конца.
    :param words:
    :return:
    """
    return u' '.join(itertools.chain([BEG_WORD, u' '], words, [ u' ', END_WORD]))


def undress(s):
    return s.replace(BEG_CHAR, u' ').replace(END_CHAR, u' ').strip()


def vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, answer_shingles,
                       premise_beg_shingles, question_beg_shingles,
                       premise_end_shingles, question_end_shingles,
                       premise_sdr, question_sdr,
                       answer_prev_chars, word_index, char_index,
                       inshingle2id, outshingle2id, outchar2id):
    ps = set(premise_shingles)
    qs = set(question_shingles)

    common_shingles = ps & qs
    notmatched_ps = ps - qs
    notmatched_qs = qs - ps

    nb_inshingles = len(inshingle2id)

    icol = 0

    sx = [common_shingles, notmatched_ps, notmatched_qs,
          premise_beg_shingles, question_beg_shingles,
          premise_end_shingles, question_end_shingles]

    for shingles in sx:
        for shingle in shingles:
            if shingle in inshingle2id:
                X_data[idata, icol+inshingle2id[shingle]] = True
        icol += nb_inshingles

    nb_outshingles = len(outshingle2id)
    for shingle in answer_shingles:
        if shingle in outshingle2id:
            X_data[idata, icol+outshingle2id[shingle]] = True
    icol += nb_outshingles

    for c in answer_prev_chars:
        X_data[idata, icol+outchar2id[c]] = True
    icol += NB_PREV_CHARS*len(outchar2id)

    X_data[idata, icol] = word_index
    icol += 1

    X_data[idata, icol] = char_index
    icol += 1

    if premise_sdr is not None:
        #for i, x in enumerate(premise_sdr):
        #    if x:
        #        X_data[idata, icol+i] = True
        X_data[idata, icol:icol+PHRASE_DIM] = premise_sdr[0, :]
        icol += PHRASE_DIM

        #for i, x in enumerate(question_sdr):
        #    if x:
        #        X_data[idata, icol+i] = True
        X_data[idata, icol:icol+PHRASE_DIM] = question_sdr[0, :]
        icol += PHRASE_DIM



def generate_answer(xgb_answer_generator, tokenizer,
                    outshingle2id, inshingle2id, outchar2id,
                    shingle_len, nb_prev_chars, nb_features, id2outchar, phrase2sdr,
                    premise, question):
    premise_words = tokenizer.tokenize(premise)
    question_words = tokenizer.tokenize(question)

    premise_wx = words2str(premise_words)
    question_wx = words2str(question_words)

    premise_shingles = ngrams(premise_wx, shingle_len)
    question_shingles = ngrams(question_wx, shingle_len)

    premise_beg_shingles = ngrams(premise_wx[:BEG_LEN], SHINGLE_LEN)
    question_beg_shingles = ngrams(question_wx[:BEG_LEN], SHINGLE_LEN)

    premise_end_shingles = ngrams(premise_wx[-END_LEN:], SHINGLE_LEN)
    question_end_shingles = ngrams(question_wx[-END_LEN:], SHINGLE_LEN)

    if phrase2sdr is not None:
        premise_sdr = phrase2sdr[premise_wx]
        question_sdr = phrase2sdr[question_wx]
    else:
        premise_sdr = None
        question_sdr = None

    answer_chain = BEG_CHAR

    while True:
        # цикл добавления новых сгенерированных символов
        answer_len = len(answer_chain)
        answer_shingles = ngrams(answer_chain, shingle_len)
        answer_prev_chars = answer_chain[max(0, answer_len - nb_prev_chars):answer_len]
        answer_prev_chars = answer_prev_chars[::-1]

        left_chars = answer_chain[1:]

        # номер генерируемого слова получаем как число пробелов слева
        word_index = left_chars.count(u' ')

        # номер генерируемого символа в генерируемом слове - отсчитываем от последнего пробела
        rpos = left_chars.rfind(u' ')
        if rpos == -1:
            # это первое слово
            char_index = len(left_chars)
        else:
            char_index = len(left_chars) - rpos - 1

        X_data = lil_matrix((1, nb_features), dtype='float')
        vectorize_sample_x(X_data, 0, premise_shingles, question_shingles, answer_shingles,
                           premise_beg_shingles, question_beg_shingles,
                           premise_end_shingles, question_end_shingles,
                           premise_sdr, question_sdr,
                           answer_prev_chars, word_index, char_index,
                           inshingle2id, outshingle2id, outchar2id)

        D_data = xgboost.DMatrix(X_data, silent=True)
        y = xgb_answer_generator.predict(D_data)
        c = id2outchar[y[0]]
        answer_chain += c
        if c == END_CHAR or answer_len >= 100:
            break

    return u'{}'.format(answer_chain[1:-1]).strip()


# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Answer text generator')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')

args = parser.parse_args()

run_mode = args.run_mode

tokenizer = Tokenizer()

config_path = os.path.join(tmp_folder,'xgb_answer_generator.config')

if run_mode == 'train':
    df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
    print('samples.count={}'.format(df.shape[0]))

    input_shingles = set()
    output_shingles = set()
    inshingle2freq = Counter()
    outshingle2freq = Counter()

    phrases1 = []

    for i, record in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Shingles and chars'):
        for phrase in [record['premise'], record['question']]:
            words = tokenizer.tokenize(phrase)
            wx = words2str(words)
            for s in ngrams(wx, SHINGLE_LEN):
                input_shingles.update(s)
                inshingle2freq[s] += 1

            phrases1.append(wx)

        answer = record['answer'].lower()
        for s in ngrams(BEG_CHAR+answer+END_CHAR, SHINGLE_LEN):
            output_shingles.update(s)
            outshingle2freq[s] += 1

    all_chars = set(itertools.chain(*phrases1))
    max_phrase_len = max(map(len, phrases1))

    nb_chars = len(all_chars)
    print('nb_chars={}'.format(nb_chars))
    print('max_phrase_len={}'.format(max_phrase_len))

    char2index = dict((c, i) for (i, c) in enumerate(filter(lambda z: z != u' ', all_chars)))

    phrase2sdr = None
    if PHRASE_DIM > 0:
        # обучаем проектор
        nb_features = nb_chars * max_phrase_len
        X_data = lil_matrix((len(phrases1), nb_features), dtype='float')
        phrase2index = dict()
        for i, phrase in enumerate(phrases1):
            phrase2index[phrase] = i
            for j, c in enumerate(phrase):
                if c in char2index:
                    X_data[i, j*nb_chars + char2index[c]] = True

        sdr_generator = None
        sdr_generator = SparseRandomProjection(n_components=PHRASE_DIM, density=0.01, eps=0.01)
        X2 = sdr_generator.fit_transform(X_data)

        # контроль sparsity получившихся векторов предложений
        n10 = X2.shape[0]*X2.shape[1]
        n1 = X2.count_nonzero()

        if False:
            n1 = 0
            ni = X2.shape[0]
            for i in tqdm.tqdm(range(ni), total=ni, desc='Calculate sparsity'):
                z = sum([int(X2[i, j] != 0.0) for j in range(PHRASE_DIM)])
                n1 += z

        print('SDR sparsity={}'.format(float(n1)/float(n10)))

        # Для фраз из списка phrases1 получаем SDR векторы в X2
        phrase2sdr = dict()
        for i, phrase in tqdm.tqdm(enumerate(phrases1), total=len(phrases1), desc='Storing SDRs'):
            #phrase2sdr[phrase] = [int(X2[i, j] != 0.0) for j in range(PHRASE_DIM)]
            phrase2sdr[phrase] = X2[i]


    # оставляем только шинглы с частотой не менее порога
    input_shingles = set(s for s, f in inshingle2freq.iteritems() if f >= MIN_SHINGLE_FREQ)
    output_shingles = set(s for s, f in outshingle2freq.iteritems() if f >= MIN_SHINGLE_FREQ)

    nb_inshingles = len(input_shingles)
    inshingle2id = dict([(s, i) for i, s in enumerate(input_shingles)])
    print('nb_inshingles={}'.format(nb_inshingles))

    nb_outshingles = len(output_shingles)
    outshingle2id = dict([(s, i) for i, s in enumerate(output_shingles)])
    print('nb_outshingles={}'.format(nb_outshingles))

    # --------------------------------------------------------------------------

    premises = []
    questions = []
    answers = []

    for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Counting'):
        premise = row['premise']
        question = row['question']
        answer = row['answer'].lower()
        if answer not in [u'да', u'нет']:
            premises.append(premise)
            questions.append(question)
            answers.append(BEG_CHAR+answer+END_CHAR)

    SEED = 123456
    TEST_SHARE = 0.2
    premises_train, premises_test,\
    questions_train, questions_test,\
    answers_train, answers_test = train_test_split(premises, questions, answers,
                                                   test_size=TEST_SHARE,
                                                   random_state=SEED)

    # оставим в разборах те, для которых все символы ответа присутствуют и в тестовом, и в
    # тренировочном наборах.
    set1 = set()
    for a in answers_train:
        set1.update(a)

    set2 = set()
    for a in answers_test:
        set2.update(a)

    missing_chars = (set1 - set2) | (set2 - set1)
    print(u'missing_chars={}'.format(u' '.join(missing_chars)))

    if len(missing_chars) > 0:
        print('Removing samples with cross-missing chars')
        premises_train0 = []
        questions_train0 = []
        answers_train0 = []
        for premise, question, answer in itertools.izip(premises_train, questions_train, answers_train):
            if not any(c in missing_chars for c in answer):
                premises_train0.append(premise)
                questions_train0.append(question)
                answers_train0.append(answer)

        premises_test0 = []
        questions_test0 = []
        answers_test0 = []
        for premise, question, answer in itertools.izip(premises_test, questions_test, answers_test):
            if not any(c in missing_chars for c in answer):
                premises_test0.append(premise)
                questions_test0.append(question)
                answers_test0.append(answer)

        premises_train = premises_train0
        questions_train = questions_train0
        answers_train = answers_train0

        premises_test = premises_test0
        questions_test = questions_test0
        answers_test = answers_test0

    nb_train = sum((len(x)+2) for x in answers_train)
    nb_test = sum((len(x)+2) for x in answers_test)

    all_outchars = set([BEG_CHAR, END_CHAR])
    for answer in itertools.chain(answers_train, answers_test):
        all_outchars.update(answer.lower())

    nb_outchars = len(all_outchars)
    outchar2id = dict([(c, i) for i, c in enumerate(all_outchars)])
    print('nb_outchars={}'.format(nb_outchars))

    print('nb_train={} nb_test={}'.format(nb_train, nb_test))

    nb_features = nb_inshingles*3 + nb_outshingles + nb_outchars*NB_PREV_CHARS
    nb_features += 2  # номер генерируемого слова и номер символа в генерируемом слове
    nb_features += nb_inshingles*2  # шинглы в начальных фрагментах предпосылки и вопроса
    nb_features += nb_inshingles*2  # шинглы в конечных фрагментах предпосылки и вопроса
    nb_features += 2*PHRASE_DIM  # SDR предпосылки и вопроса

    print('nb_features={}'.format(nb_features))

    X_train = lil_matrix((nb_train, nb_features), dtype='float32')
    y_train = []

    X_test = lil_matrix((nb_test, nb_features), dtype='float32')
    y_test = []

    for train_or_test in range(2):
        if train_or_test == 0:
            premises = premises_train
            questions = questions_train
            answers = answers_train
            X_data = X_train
            y_data = y_train
            descr = 'Vectorization of training set'
        else:
            premises = premises_test
            questions = questions_test
            answers = answers_test
            X_data = X_test
            y_data = y_test
            descr = 'Vectorization of test data'

        idata = 0

        for index, (premise, question, answer) in tqdm.tqdm(enumerate(itertools.izip(premises, questions, answers)),
                                                            total=len(premises),
                                                            desc=descr):
            premise_words = tokenizer.tokenize(premise)
            question_words = tokenizer.tokenize(question)

            premise_shingles = ngrams(words2str(premise_words), SHINGLE_LEN)
            question_shingles = ngrams(words2str(question_words), SHINGLE_LEN)

            premise_beg_shingles = ngrams(words2str(premise_words)[:BEG_LEN], SHINGLE_LEN)
            question_beg_shingles = ngrams(words2str(question_words)[:BEG_LEN], SHINGLE_LEN)

            premise_end_shingles = ngrams(words2str(premise_words)[-END_LEN:], SHINGLE_LEN)
            question_end_shingles = ngrams(words2str(question_words)[-END_LEN:], SHINGLE_LEN)

            if phrase2sdr is not None:
                premise_sdr = phrase2sdr[words2str(premise_words)]
                question_sdr = phrase2sdr[words2str(question_words)]
            else:
                premise_sdr = None
                question_sdr = None

            answer2 = answer
            for answer_len in range(1, len(answer2)):
                answer_chain = answer2[:answer_len] # эта цепочка уже сгенерирована к данному моменту
                answer_shingles = ngrams(answer_chain, SHINGLE_LEN)
                next_char = answer2[answer_len]
                answer_prev_chars = answer2[max(0, answer_len-NB_PREV_CHARS):answer_len]
                answer_prev_chars = answer_prev_chars[::-1] # чтобы предпоследний символ был всегда на фиксированном месте в метрице, etc

                left_chars = answer2[1:answer_len]

                # номер генерируемого слова получаем как число пробелов слева
                word_index = left_chars.count(u' ')

                # номер генерируемого символа в генерируемом слове - отсчитываем от последнего пробела
                rpos = left_chars.rfind(u' ')
                if rpos == -1:
                    # это первое слово
                    char_index = len(left_chars)
                else:
                    char_index = len(left_chars) - rpos - 1

                vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, answer_shingles,
                                   premise_beg_shingles, question_beg_shingles,
                                   premise_end_shingles, question_end_shingles,
                                   premise_sdr, question_sdr,
                                   answer_prev_chars, word_index, char_index,
                                   inshingle2id, outshingle2id, outchar2id)
                y_data.append(outchar2id[next_char])

                idata += 1

    if X_train.shape[0] != len(y_train):
        X_train = X_train[0:len(y_train), :]

    if X_test.shape[0] != len(y_test):
        X_test = X_test[0:len(y_test), :]

    id2outchar = dict([(i, c) for c, i in outchar2id.items()])

    print('uniques(y_train)={}'.format(len(set(y_train))))
    print('uniques(y_test)={}'.format(len(set(y_test))))

    for y in set(y_train)-set(y_test):
        c = id2outchar[y]
        print(u'Missing in y_test: {}'.format(c))

    D_train = xgboost.DMatrix(X_train, y_train, silent=0)
    D_val = xgboost.DMatrix(X_test, y_test, silent=0)

    xgb_params = {
        'booster': 'gbtree',
        'subsample': 1.0,
        'max_depth': MAX_DEPTH,
        'seed': 123456,
        'min_child_weight': 1,
        'eta': 0.45,
        'gamma': 0.01,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'eval_metric': 'merror',
        'objective': 'multi:softmax',
        'num_class': nb_outchars,  # len(set(y_train)),
        'silent': 1,
        # 'updater': 'grow_gpu'
    }

    print('Train model...')
    cl = xgboost.train(xgb_params,
                       D_train,
                       evals=[(D_val, 'val')],
                       num_boost_round=NB_TREES,
                       verbose_eval=10,
                       early_stopping_rounds=10)

    print('Training is finished')

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_filename = os.path.join(tmp_folder, 'xgb_answer_generator.model')
    model_config = {
                    'solver': 'xgb',
                    'outshingle2id': outshingle2id,
                    'inshingle2id': inshingle2id,
                    'outchar2id': outchar2id,
                    'model_filename': model_filename,
                    'shingle_len': SHINGLE_LEN,
                    'NB_PREV_CHARS': NB_PREV_CHARS,
                    'BEG_LEN': BEG_LEN,
                    'nb_features': nb_features,
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    cl.save_model(model_filename)

    # Финальные оценки точности

    y_pred = cl.predict(D_val)
    y_pred = (y_pred >= 0.5).astype(np.int)
    acc = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    print('per char accuracy={}'.format(acc))

    nb_errors = 0
    nb_test = min(1000, nb_test)
    for premise, question, answer in tqdm.tqdm(itertools.izip(premises_test[:nb_test],
                                                              questions_test[:nb_test],
                                                              answers_test[:nb_test]),
                                               total=nb_test,
                                               desc='Calculating instance accuracy'):
        answer2 = generate_answer(cl, tokenizer,
                                  outshingle2id, inshingle2id, outchar2id,
                                  SHINGLE_LEN, NB_PREV_CHARS, nb_features, id2outchar, phrase2sdr,
                                  premise, question)

        if undress(answer2) != undress(answer):
            nb_errors += 1
            #print('(-) ', end='')
        #else:
            #print('(+) ', end='')
        #print(u'true={} model={}'.format(undress(answer), undress(answer2)))

    print('per instance accuracy={}'.format(float(nb_test-nb_errors)/float(nb_test)))


if run_mode == 'query':

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    outshingle2id = cfg['outshingle2id']
    inshingle2id = cfg['inshingle2id']
    outchar2id = cfg['outchar2id']
    model_filename = cfg['model_filename']
    SHINGLE_LEN = cfg['shingle_len']
    NB_PREV_CHARS = cfg['NB_PREV_CHARS']
    BEG_LEN = cfg['BEG_LEN']
    nb_features = cfg['nb_features']

    generator = xgboost.Booster()
    generator.load_model(cfg['model_filename'])

    phrase2sdr = None

    id2outchar = dict([(i, c) for c, i in outchar2id.items()])

    while True:
        premise = raw_input('Premise:> ').decode(sys.stdout.encoding).strip().lower()
        if len(premise) == 0:
            break

        question = raw_input('Question:> ').decode(sys.stdout.encoding).strip().lower()
        if len(question) == 0:
            break

        answer = generate_answer(generator, tokenizer,
                                 outshingle2id, inshingle2id, outchar2id,
                                 SHINGLE_LEN, NB_PREV_CHARS, nb_features, id2outchar, phrase2sdr,
                                 premise, question)

        print(u'{}'.format(answer))




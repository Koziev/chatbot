# -*- coding: utf-8 -*-
'''
Тренировка модели, которая посимвольно в режиме teacher forcing учится генерировать
ответ для заданной предпосылки и вопроса.

В качестве классификационного движка для выбора символов используется XGBoost.

За один запуск модели выбирается один новый символ, который добавляется к ранее сгенерированной
цепочке символов ответа (см. функцию generate_answer).
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import itertools
import json
import os
import sys
import argparse
import codecs
import gzip
from collections import Counter
import six
import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers

# основной настроечный метапараметр - длина символьных шинглов для представления
# предпосылки и вопроса (bag of shingles).
SHINGLE_LEN = 3

NB_PREV_CHARS = 5 # кол-во пред. символов в сгенерированном ответе, учитываемых при выборе следующего символа.

NB_SAMPLES = 1000000 # кол-во записей в датасете (до разбивки на тренировку и валидацию)

# Шинглы с частотой меньше указанной не будут давать входные фичи.
MIN_SHINGLE_FREQ = 2

BEG_LEN = 10  # длина в символах начального фрагмента фраз, который дает отдельные фичи
END_LEN = 10  # длина в символах конечного фрагмента фраз, который дает отдельные фичи

NB_TREES = 10000
MAX_DEPTH = 8  # макс. глубина для градиентного бустинга


# -------------------------------------------------------------------

BEG_WORD = u'\b'
END_WORD = u'\n'

BEG_CHAR = u'\b'
END_CHAR = u'\n'

# -------------------------------------------------------------------


class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer



def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    """
    Цепочку слов соединяем в строку, добавляя перед цепочкой и после нее
    пробел и специальные символы начала и конца.
    :param words:
    :return:
    """
    return BEG_WORD + u' ' + u' '.join(words) + u' ' + END_WORD


def undress(s):
    return s.replace(BEG_CHAR, u' ').replace(END_CHAR, u' ').strip()


def encode_char(c):
    if c == BEG_CHAR:
        return u'\\b'
    elif c == END_CHAR:
        return u'\\r'
    else:
        return c


def vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, answer_shingles,
                       premise_beg_shingles, question_beg_shingles,
                       premise_end_shingles, question_end_shingles,
                       premise_sdr, question_sdr,
                       answer_prev_chars, word_index, char_index,
                       premise_str, premise_words,
                       question_str, question_words,
                       lexicon,
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

    # помечаем символы, которые могут быть после последнего символа в сгенерированной
    # части ответа с точки зрения строки предпосылки, вопроса и т.д.
    prev_char1 = answer_prev_chars[::-1][-1]

    premise_str1 = premise_str.replace(BEG_CHAR+u' ', BEG_CHAR)
    for c, char_index in outchar2id.items():
        if prev_char1+c in premise_str1:
            X_data[idata, icol+char_index] = True
    icol += len(outchar2id)

    question_str1 = question_str.replace(BEG_CHAR+u' ', BEG_CHAR)
    for c, char_index in outchar2id.items():
        if prev_char1+c in question_str1:
            X_data[idata, icol+char_index] = True
    icol += len(outchar2id)

    premise_words_2grams = set()
    for premise_word in premise_words:
        for wordform in lexicon.get_forms(premise_word):
            premise_words_2grams.update(ngrams(u' '+wordform+u' ', 2))
    for c, char_index in outchar2id.items():
        if prev_char1+c in premise_words_2grams:
            X_data[idata, icol+char_index] = True
    icol += len(outchar2id)

    question_words_2grams = set()
    for question_word in question_words:
        for wordform in lexicon.get_forms(question_word):
            question_words_2grams.update(ngrams(u' '+wordform+u' ', 2))
    for c, char_index in outchar2id.items():
        if prev_char1+c in question_words_2grams:
            X_data[idata, icol+char_index] = True
    icol += len(outchar2id)



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
                           premise_wx, premise_words,
                           question_wx, question_words,
                           lexicon,
                           inshingle2id, outshingle2id, outchar2id)

        D_data = xgboost.DMatrix(X_data, silent=True)
        y = xgb_answer_generator.predict(D_data)
        c = id2outchar[y[0]]
        answer_chain += c
        if c == END_CHAR or answer_len >= 100:
            break

    return u'{}'.format(answer_chain[1:-1]).strip()


class Word2Lemmas(object):
    def __init__(self):
        pass

    def load(self, path):
        print('Loading lexicon from {}'.format(path))
        self.lemmas = dict()
        self.forms = dict()
        with gzip.open(path, 'r') as rdr:
            for line in rdr:
                tx = line.strip().decode('utf8').split('\t')
                if len(tx) == 2:
                    form = tx[0]
                    lemma = tx[1]

                    if form not in self.forms:
                        self.forms[form] = [lemma]
                    else:
                        self.forms[form].append(lemma)

                    if lemma not in self.lemmas:
                        self.lemmas[lemma] = {form}
                    else:
                        self.lemmas[lemma].add(form)
        print('Lexicon loaded: {} lemmas, {} wordforms'.format(len(self.lemmas), len(self.forms)))

    def get_forms(self, word):
        if word in self.forms:
            #result = set()
            #for lemma in self.forms[word]:
            #    result.update(self.lemmas[lemma])
            #return result
            return set(itertools.chain(*(self.lemmas[lemma] for lemma in self.forms[word])))
        else:
            return [word]


# -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Answer text generator')
parser.add_argument('--run_mode', type=str, default='train', help='what to do: train | query')
parser.add_argument('--input', type=str, default='../data/pqa_all.dat', help='training dataset path')
parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')
parser.add_argument('--word2lemmas', type=str, default='../data/ru_word2lemma.tsv.gz')


args = parser.parse_args()

run_mode = args.run_mode
tmp_folder = args.tmp
data_folder = args.data_dir

# Этот датасет создается скриптом prepare_qa_dataset.py
input_path = args.input

# Отбор и упаковка пар словоформа-лемма выполняется скриптом prepare_word2lemmas.py
word2lemmas_path = args.word2lemmas

tokenizer = Tokenizer()

lexicon = Word2Lemmas()
lexicon.load(word2lemmas_path)
#for w in lexicon.get_forms(u'дяди'):
#    print(u'{}'.format(w))

config_path = os.path.join(tmp_folder,'xgb_answer_generator.config')

if run_mode == 'train':
    input_shingles = set()
    output_shingles = set()
    inshingle2freq = Counter()
    outshingle2freq = Counter()

    phrases1 = []

    # Загружаем датасет, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
    print(u'Loading samples from {}'.format(input_path))
    samples0 = []
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    tokenizer = Tokenizer()

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

                        for phrase in itertools.chain(premises, [question]):
                            words = tokenizer.tokenize(phrase)
                            wx = words2str(words)
                            phrases1.append(wx)
                            for s in ngrams(wx, SHINGLE_LEN):
                                input_shingles.add(s)
                                inshingle2freq[s] += 1

                        for s in ngrams(BEG_CHAR + answer + END_CHAR, SHINGLE_LEN):
                            output_shingles.add(s)
                            outshingle2freq[s] += 1

                    lines = []

            else:
                lines.append(line)

    samples = samples0

    all_chars = set(itertools.chain(*phrases1))
    max_phrase_len = max(map(len, phrases1))

    nb_chars = len(all_chars)
    print('nb_chars={}'.format(nb_chars))
    print('max_phrase_len={}'.format(max_phrase_len))

    char2index = dict((c, i) for (i, c) in enumerate(filter(lambda z: z != u' ', all_chars)))

    # оставляем только шинглы с частотой не менее порога
    input_shingles = set(s for s, f in inshingle2freq.iteritems() if f >= MIN_SHINGLE_FREQ)
    output_shingles = set(s for s, f in outshingle2freq.iteritems() if f >= MIN_SHINGLE_FREQ)

    nb_inshingles = len(input_shingles)
    inshingle2id = dict((s, i) for i, s in enumerate(input_shingles))
    print('nb_inshingles={}'.format(nb_inshingles))

    nb_outshingles = len(output_shingles)
    outshingle2id = dict((s, i) for i, s in enumerate(output_shingles))
    print('nb_outshingles={}'.format(nb_outshingles))

    # --------------------------------------------------------------------------

    premises = []
    questions = []
    answers = []

    for sample in samples:
        premise = sample.premises[0] if len(sample.premises) > 0 else u''
        question = sample.question
        answer = sample.answer
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
    outchar2id = dict((c, i) for i, c in enumerate(all_outchars))
    print('nb_outchars={}'.format(nb_outchars))

    print('nb_train={} nb_test={}'.format(nb_train, nb_test))

    nb_features = nb_inshingles*3 + nb_outshingles + nb_outchars*NB_PREV_CHARS
    nb_features += 2  # номер генерируемого слова и номер символа в генерируемом слове
    nb_features += nb_inshingles*2  # шинглы в начальных фрагментах предпосылки и вопроса
    nb_features += nb_inshingles*2  # шинглы в конечных фрагментах предпосылки и вопроса
    nb_features += nb_outchars  # какие символы в предпосылке бывают после текущего символа
    nb_features += nb_outchars  # какие символы бывают в любых формах слов предпосылки
    nb_features += nb_outchars  # какие символы в вопросе бывают после текущего символа
    nb_features += nb_outchars  # какие символы бывают в любых формах слов вопроса

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

            premise_str = words2str(premise_words)
            question_str = words2str(question_words)

            premise_shingles = ngrams(premise_str, SHINGLE_LEN)
            question_shingles = ngrams(words2str(question_words), SHINGLE_LEN)

            premise_beg_shingles = ngrams(words2str(premise_words)[:BEG_LEN], SHINGLE_LEN)
            question_beg_shingles = ngrams(words2str(question_words)[:BEG_LEN], SHINGLE_LEN)

            premise_end_shingles = ngrams(words2str(premise_words)[-END_LEN:], SHINGLE_LEN)
            question_end_shingles = ngrams(words2str(question_words)[-END_LEN:], SHINGLE_LEN)

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

                vectorize_sample_x(X_data, idata,
                                   premise_shingles, question_shingles, answer_shingles,
                                   premise_beg_shingles, question_beg_shingles,
                                   premise_end_shingles, question_end_shingles,
                                   premise_sdr, question_sdr,
                                   answer_prev_chars, word_index, char_index,
                                   premise_str, premise_words,
                                   question_str, question_words,
                                   lexicon,
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


    # DEBUG BEGIN
    if False:
        with codecs.open('../tmp/cy.txt', 'w', 'utf-8') as wrt:
            cy = Counter()
            cy.update(y_test)
            for y, n in cy.most_common():
                wrt.write(u'{:3d} {:2s} {}\n'.format(y, encode_char(id2outchar[y]), n))
    # DEBUG END

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
                       num_boost_round=5000,  #NB_TREES,
                       verbose_eval=10,
                       early_stopping_rounds=20)

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
                    'END_LEN': END_LEN,
                    'nb_features': nb_features,
                    'word2lemmas_path': word2lemmas_path
                   }

    with open(config_path, 'w') as f:
        json.dump(model_config, f)

    cl.save_model(model_filename)

    # Финальные оценки точности.
    y_pred = cl.predict(D_val)
    acc = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    print('per char accuracy={}'.format(acc))

    # Накопим кол-во ошибок и сэмплов для ответов разной длины.
    answerlen2samples = Counter()
    answerlen2errors = Counter()

    nb_errors = 0
    nb_test = min(1000, nb_test)
    for premise, question, answer in tqdm.tqdm(itertools.izip(premises_test[:nb_test],
                                                              questions_test[:nb_test],
                                                              answers_test[:nb_test]),
                                               total=nb_test,
                                               desc='Calculating instance accuracy'):
        answer2 = generate_answer(cl, tokenizer,
                                  outshingle2id, inshingle2id, outchar2id,
                                  SHINGLE_LEN, NB_PREV_CHARS, nb_features, id2outchar, None,
                                  premise, question)
        answer_len = len(answer)
        answerlen2samples[answer_len] += 1
        if undress(answer2) != undress(answer):
            nb_errors += 1
            answerlen2errors[answer_len] += 1

    print('per instance accuracy={}'.format(float(nb_test-nb_errors)/float(nb_test)))

    report_path = os.path.join(tmp_folder, 'xgb_answer_generator.report.txt')
    with codecs.open(report_path, 'w', 'utf-8') as wrt:
        wrt.write(u'Accuracy for answers with respect to their lengths:\n')
        for answer_len in sorted(answerlen2samples.keys()):
            support = answerlen2samples[answer_len]
            nb_err = answerlen2errors[answer_len]
            acc = 1.0 - float(nb_err)/float(support)
            wrt.write(u'{:3d} {}\n'.format(answer_len, acc))

        wrt.write('\n\n')
        wrt.write('Multiclass classification report:\n')
        # Для classification_report нужен список только тех названий классов, которые
        # встречаются в y_test, иначе получим неверный отчет и ворнинг в придачу.
        class_names = [encode_char(id2outchar[y]) for y in sorted(set(y_test) | set(y_pred))]
        wrt.write(classification_report(y_test, y_pred, target_names=class_names))
        wrt.write('\n\n')
        #wrt.write(confusion_matrix(y_test, y_pred, labels=class_names))

    # Accuracy for answers with respect to their lengths:
    with open(os.path.join(tmp_folder, 'xgb_answer_generator.accuracy.csv'), 'w') as wrt:
        wrt.write('answer_len\tnb_samples\taccuracy\n')
        for answer_len in sorted(answerlen2samples.keys()):
            support = answerlen2samples[answer_len]
            nb_err = answerlen2errors[answer_len]
            acc = 1.0 - float(nb_err)/float(support)
            wrt.write(u'{}\t{}\t{}\n'.format(answer_len, support, acc))



if run_mode == 'query':
    # Ручное тестирование натренированной модели генерации ответа.
    # Сначала загружаем результаты тренировки.
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

    id2outchar = dict((i, c) for c, i in outchar2id.items())

    while True:
        premise = u''
        question = None

        premise = raw_input('Premise:> ').decode(sys.stdout.encoding).strip().lower()
        if len(premise) > 0 and premise[-1] == u'?':
            question = premise
            premise = u''

        if question is None:
            question = raw_input('Question:> ').decode(sys.stdout.encoding).strip().lower()
            if len(question) == 0:
                break

        answer = generate_answer(generator, tokenizer,
                                 outshingle2id, inshingle2id, outchar2id,
                                 SHINGLE_LEN, NB_PREV_CHARS, nb_features, id2outchar, phrase2sdr,
                                 premise, question)

        print(u'Answer: {}'.format(answer))

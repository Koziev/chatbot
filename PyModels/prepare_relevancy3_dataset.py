# -*- coding: utf-8 -*-
'''
Готовим датасет для обучения модели с архитектурой triple loss, определяющей
релевантность вопроса и предпосылки. Создаются тройки (предпосылка, релевантный вопрос, нерелевантный вопрос)

Для проекта чатбота (https://github.com/Koziev/chatbot)

(c) by Koziev Ilya inkoziev@gmail.com
'''

from __future__ import division
from __future__ import print_function

import codecs
import collections
import itertools
import os
import random
import numpy as np

try:
    from itertools import izip as zip
except ImportError:
    pass

from utils.tokenizer import Tokenizer


# Путь к создаваемому датасету для модели детектора на базе triplet loss
output_filepath3 = '../data/relevancy_dataset3.csv'


USE_AUTOGEN = True  # добавлять ли сэмплы из автоматически сгенерированных датасетов

HANDCRAFTED_WEIGHT = 1 # вес для сэмплов, которые в явном виде созданы вручную
AUTOGEN_WEIGHT = 1 # вес для синтетических сэмплов, сгенерированных автоматически

# Автоматически сгенерированных сэмплов очень много, намного больше чем вручную
# составленных, поэтому ограничим количество паттернов для каждого типа автоматически
# сгенерированных.
MAX_NB_AUTOGEN = 100000  # макс. число автоматически сгенерированных сэмплов одного типа

ADD_PARAPHRASES = False

n_negative_per_positive = 1

tmp_folder = '../tmp'
data_folder = '../data'
paraphrases_paths = ['../data/paraphrases.txt', '../data/contradictions.txt']
qa_paths = [('qa.txt', HANDCRAFTED_WEIGHT, 10000000)]

if USE_AUTOGEN:
    qa_paths.extend([('current_time_pqa.txt', AUTOGEN_WEIGHT, 1000000),
                     ('premise_question_answer6.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer5.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer4.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer4_1s.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer4_2s.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer5_1s.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer5_2s.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                    ])
questions_path = '../data/questions.txt'

stop_words = set(u'не ни ль и или ли что какой же ж какая какие сам сама сами само был были было есть '.split())
stop_words.update(u'о а в на у к с со по ко мне нам я он она над за из от до'.split())

# ---------------------------------------------------------------


class Sample3:
    def __init__(self, anchor, positive, negative):
        assert(len(anchor) > 0)
        assert(len(positive) > 0)
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

    def key(self):
        return self.anchor + u'|' + self.positive + u'|' + self.negative


def ru_sanitize(s):
    return s.replace(u'ё', u'е')


def normalize_qline(line):
    line = line.replace(u'(+)', u'')
    line = line.replace(u'(-)', u'')
    line = line.replace(u'T:', u'')
    line = line.replace(u'Q:', u'')
    line = line.replace(u'A:', u'')
    line = line.replace(u'\t', u' ')
    line = line.replace('.', ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ').replace('-', ' ')
    line = line.replace('  ', ' ').strip().lower()
    line = ru_sanitize(line)
    return line


def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(shingles1, shingles2):
    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2))


# ---------------------------------------------------------------

tokenizer = Tokenizer()

# ------------------------------------------------------------------------

random_questions = []

qwords = set(u'кто кому ком когда где зачем почему откуда куда как сколько что чем чему чего'.split())

# прочитаем список случайных вопросов из заранее сформированного файла
# (см. код на C# https://github.com/Koziev/chatbot/tree/master/CSharpCode/ExtractFactsFromParsing
# и результаты его работы https://github.com/Koziev/NLP_Datasets/blob/master/Samples/questions4.txt)
print('Loading random questions from {}'.format(questions_path))
with codecs.open(questions_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        if len(line) < 40:
            question = line.strip()
            words = tokenizer.tokenize(question)
            if any((w in qwords) for w in words):
                question = ru_sanitize(u' '.join(words))
                random_questions.append(question)

for facts_path in ['facts4.txt', 'facts5.txt', 'facts6.txt']:
    print('Loading random questions from {}'.format(facts_path))
    with codecs.open(os.path.join(data_folder, facts_path), 'r', 'utf-8') as rdr:
        n = 0
        for line in rdr:
            if n > 100000:
                s = line.strip()
                if s[-1] == u'?':
                    words = tokenizer.tokenize(question)
                    if any((w in qwords) for w in words):
                        question = ru_sanitize(u' '.join(words))
                        random_questions.append(question)
            n += 1
            if n > 2000000:
                break


for q_path in qa_paths:
    print('Loading random questions from {}'.format(q_path))
    with codecs.open(os.path.join(data_folder, q_path[0]), 'r', 'utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s.startswith('Q:'):
                question = normalize_qline(s)
                words = tokenizer.tokenize(question)
                if any((w in qwords) for w in words):
                    question = u' '.join(words)
                    random_questions.append(question)

random_questions = list(random_questions)

print('Prepare index of {} random questions...'.format(len(random_questions)))
word2random_questions = dict()
for iquest, quest in enumerate(random_questions):
    words = tokenizer.tokenize(quest)

    for word in words:
        if word not in stop_words:
            if word not in word2random_questions:
                word2random_questions[word] = [iquest]
            else:
                word2random_questions[word].append(iquest)

print('{} random questions in set'.format(len(random_questions)))

# Для генерации негативных паттернов нам надо будет для каждого
# предложения быстро искать близкие к нему по критерию Жаккара.
# Заранее подготовим списки шинглов для датасета "qa.txt".
shingle_len = 3
tokenizer = Tokenizer()
phrases1 = []  # список кортежей (предпосылка, слова_без_повторов, шинглы)
with codecs.open(os.path.join(data_folder, 'qa.txt'), 'r', 'utf-8') as rdr:
    for phrase in rdr:
        if phrase.startswith(u'T:'):
            phrase = phrase.replace(u'T:', u'').lower().strip()
            phrase = ru_sanitize(phrase)
            words = tokenizer.tokenize(phrase.strip())
            if len(words) > 0:
                shingles = ngrams(u' '.join(words), shingle_len)
                phrases1.append((u' '.join(words), set(words), shingles))


# ------------------------------------------------------------------------
# Генерируем датасет из троек (вопрос, релевантная_предпосылка, нерелевантная_предпосылка)
# для модели с triple loss.

# Сначала здесь накопим пары релевантных предпосылок и вопросов
samples2 = []

# Тут будем сохранять все добавленные пары предпосылка+вопрос.
# Это поможет исключить дубликаты, а также предотвратит добавление
# негативных пар, которые уже известны как позитивные.
all_pq = set()

lines = []
random_questions2 = set()

# Из отдельного файла загрузим список нерелевантных пар предпосылка-вопрос.
manual_negatives = dict()
with codecs.open(os.path.join(data_folder, 'nonrelevant_premise_questions.txt'), 'r', 'utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        if line:
            tx = line.split('|')
            if len(tx) == 2:
                premise = normalize_qline(tx[0])
                question = normalize_qline(tx[1])
                if premise not in manual_negatives:
                    manual_negatives[premise] = [question]
                else:
                    manual_negatives[premise].append(question)


# Собираем список релевантных пар предпосылка-вопрос
for qa_path, qa_weight, max_samples in qa_paths:
    print('Parsing {}'.format(qa_path))
    nsample = 0
    with codecs.open(os.path.join(data_folder, qa_path), "r", "utf-8") as inf:
        loading_state = 'T'

        text = []
        questions = []

        for line in inf:
            if nsample > max_samples:
                break

            line = line.strip()

            if line.startswith(u'T:'):
                if loading_state == 'T':
                    text.append(normalize_qline(line))
                else:
                    # закончился парсинг предыдущего блока из текста (предпосылки),
                    # вопросов и ответов.
                    # Из загруженных записей добавим пары в обучающий датасет
                    if len(text) == 1:
                        #premise_questions.append((text[0], questions))

                        for premise in text:
                            for question in questions:
                                random_questions2.add(question)
                                pq = premise + u'|' + question
                                if pq not in all_pq:
                                    samples2.append((premise, question))
                                    all_pq.add(pq)
                                    nsample += 1

                    loading_state = 'T'
                    questions = []
                    text = [normalize_qline(line)]

            elif line.startswith(u'Q:'):
                loading_state = 'Q'
                questions.append(normalize_qline(line))

# Добавляем перефразировки
if ADD_PARAPHRASES:
    lines = []
    for paraphrases_path in paraphrases_paths:
        print('Parsing {} '.format(paraphrases_path))
        posit_pairs_count1 = 0  # кол-во релевантных пар, извлеченных из обрабатываемого файла
        negat_pairs_count1 = 0  # кол-во нерелевантных пар, извлеченных из обрабатываемого файла
        random_negat_pairs_count1 = 0  # кол-во добавленных случайных нерелевантных пар
        with codecs.open(paraphrases_path, "r", "utf-8") as inf:
            for line in inf:
                line = line.strip()
                if len(line) == 0:
                    if len(lines) > 1:
                        posit_lines = []
                        negat_lines = []
                        for line in lines:
                            if line.startswith(u'(+)') or not line.startswith(u'(-)'):
                                posit_lines.append(normalize_qline(line))
                            else:
                                negat_lines.append(normalize_qline(line))

                        for i1 in range(len(posit_lines)):
                            for i2 in range(len(posit_lines)):
                                str1 = posit_lines[i1]
                                str2 = posit_lines[i2]
                                if str1 != str2:
                                    pq = str1 + u'|' + str2
                                    if pq not in all_pq:
                                        samples2.append((str1, str2))
                                        all_pq.add(pq)

                                    posit_pairs_count1 += 1

                    lines = []
                else:
                    lines.append(line)


# Второй проход по списку пар предпосылка-вопрос.
# Для каждой пары подбираем негативный вопрос.
random_questions3 = list(random_questions2)
samples3 = []
n3 = 0
for premise, question in samples2:
    # Для предпосылки есть заданные вручную негативные вопросы?
    if premise in manual_negatives:
        for nonrelevant_question in manual_negatives[premise]:
            samples3.append(Sample3(premise, question, nonrelevant_question))
            pq = premise + u'|' + nonrelevant_question
            all_pq.add(pq)
    else:
        # Рандомно генерируем нерелевантный вопрос.
        for _ in range(n_negative_per_positive):
            # случайные вопросы, имеющие общие слова с предпосылкой
            # таким образом, получаются негативные сэмплы типа "кошка ловит мышей":"мышей в амбаре нет?"
            words = tokenizer.tokenize(premise)
            selected_random_questions = None
            for word in words:
                if word not in stop_words:
                    if word in word2random_questions:
                        if selected_random_questions is None:
                            selected_random_questions = set(word2random_questions[word])
                        else:
                            selected_random_questions |= set(word2random_questions[word])

            n_added_neg = 0
            if selected_random_questions is not None and len(selected_random_questions) > 0:
                words = set(words)
                selected_random_questions = list(selected_random_questions)
                selected_random_questions = np.random.permutation(selected_random_questions)
                if len(selected_random_questions) > n_negative_per_positive:
                    selected_random_questions = selected_random_questions[:n_negative_per_positive]
                    for random_question in selected_random_questions:
                        random_question = random_questions[random_question]
                        pq = premise + u'|' + random_question
                        if pq not in all_pq:
                            samples3.append(Sample3(premise, question, random_question))
                            n_added_neg += 1

            for _ in range(max(0, n_negative_per_positive - n_added_neg)):
                # абсолютно случайный вопрос
                while True:
                    random_question = random.choice(random_questions3)
                    pq = premise+u'|'+random_question
                    if pq not in all_pq:
                        samples3.append(Sample3(premise, question, random_question))
                        break

    # эксперимент от 15.11.2018
    if False:
        qx = [u'как меня зовут', u'кто я', u'где ты находишься', u'какой я']
        for nonrelevant_question in qx:
            pq = premise + u'|' + nonrelevant_question
            if pq not in all_pq:
                samples3.append(Sample3(premise, question, nonrelevant_question))
                all_pq.add(pq)


print(u'Storing {} triplets to dataset "{}"'.format(len(samples3), output_filepath3))
with codecs.open(output_filepath3, 'w', 'utf-8') as wrt:
    wrt.write(u'anchor\tpositive\tnegative\n')
    for sample in samples3:
        wrt.write(u'{}\t{}\t{}\n'.format(sample.anchor, sample.positive, sample.negative))


# -*- coding: utf-8 -*-
"""
Парсим файл с текстами, вопросами и ответами, готовим датасет
для тренировки моделей генерации текста ответа из пары предпосылка+вопрос.
(c) by Koziev Ilya inkoziev@gmail.com
"""

from __future__ import division
from __future__ import print_function

import codecs
import os
import tqdm
import io
import random

import pandas as pd
import csv
import sklearn.utils

from ruchatbot.utils.tokenizer import Tokenizer

tmp_folder = '../../tmp'
data_folder = '../../data'
paraphrases_path = 'paraphrases.txt'
qa_path = 'qa.txt'
contradictions_path = 'contradictions.txt'

USE_AUTOGEN = True  # добавлять ли сэмплы из автоматически сгенерированных датасетов
MAX_AUTOGEN_SAMPLES = 100000

# ---------------------------------------------------------------

class Sample:
    def __init__(self, premises, question, answer):
        self.premises = premises[:]
        self.question = question
        self.answer = answer

    def save(self, wrt):
        for premise in self.premises:
            wrt.write(u'{}\n'.format(premise))
        wrt.write(u'{}\n'.format(self.question))
        wrt.write(u'{}\n'.format(self.answer))


def ru_sanitize(s):
    return s.replace(u'ё', u'е')


def normalize_qline(line):
    line = line.replace(u'(+)', u'')
    line = line.replace(u'(-)', u'')
    line = line.replace(u'T:', u'')
    line = line.replace(u'Q:', u'')
    line = line.replace(u'A:', u'')
    line = line.replace(u'\t', u' ')
    line = line.replace('.', ' ').replace('?', ' ').replace('!', ' ')
    line = line.replace('  ', ' ').strip().lower()
    line = ru_sanitize(line)
    return line


# ---------------------------------------------------------------

result_path = os.path.join(data_folder, 'premise_question_answer.csv')
pqa_yesno_path = os.path.join(data_folder, 'pqa_yes_no.dat')
pqa_all_path = os.path.join(data_folder, 'pqa_all.dat')


tokenizer = Tokenizer()
tokenizer.load()

records = [] # список из (предпосылка, вопрос, ответ, паттерн_создан_вручную)

added_records_set = set() # для предотвращения повторов


def add_record(premise, question, answer, is_handmade):
    premise = premise.strip()
    question = question.strip()
    answer = answer.strip()

    if not premise or not question or not answer:
        print(u'ERROR empty phrase in: premise={} question={} answer={}'.format(premise, question, answer))
        raise ValueError()

    s1 = ru_sanitize(premise)
    s2 = ru_sanitize(question)
    s3 = ru_sanitize(answer)

    k = s1+u'|'+s2+u'|'+s3
    if k not in added_records_set:
        added_records_set.add(k)
        records.append((s1, s2, s3, is_handmade))

# ------------------------------------------------------------------------

if True:
    # Добавляем перефразировочные пары как сэмплы с ответом 'да'
    print('Parsing {:<40s} '.format(paraphrases_path), end='')

    question_words = set(u'кто что кому чему кем чем ком чем почему зачем когда где сколько откуда куда'.split())

    nb_paraphrases = 0
    lines = []
    with io.open(os.path.join(data_folder, paraphrases_path), "r", encoding="utf-8") as inf:
        for line in inf:
            line = line.strip()
            if len(line) == 0:
                if len(lines) > 1:
                    questions = []
                    premises = []
                    for line in lines:
                        if line.startswith(u'(+)') or not line.startswith(u'(-)'):
                            s = normalize_qline(line)
                            if line[-1] == u'?':
                                questions.append(s)
                            else:
                                premises.append(s)
                                questions.append(s)

                    for premise in premises:
                        premise_words = tokenizer.tokenize(premise)
                        if any((word in question_words) for word in premise_words):
                            # Пропускаем фразу, так как это вопрос, и в качестве предпосылки он не будет использоваться
                            continue

                        for question in questions:
                            if random.random() < 0.1:
                                add_record(premise, question, u'да', False)
                                nb_paraphrases += 1

                lines = []
            else:
                lines.append(line)

    print('{:<6d} patterns'.format(nb_paraphrases))

# ------------------------------------------------------------------------

# 1) Используем созданные вручную паттерны из qa_path
# 2) Добавляем автоматически извлеченные паттерны
px = [(qa_path, True)]

if USE_AUTOGEN:
    px2 = ['current_time_pqa.txt',
            'names_pqa.txt',
            'premise_question_answer_names4.txt',
            'premise_question_answer_neg4.txt',
            'premise_question_answer_neg4_1s.txt',
            'premise_question_answer_neg4_2s.txt',
            'premise_question_answer_neg5.txt',
            'premise_question_answer_neg5_1s.txt',
            'premise_question_answer_neg5_2s.txt',
            'premise_question_answer6.txt',
            'premise_question_answer5.txt',
            'premise_question_answer4.txt',
            'premise_question_answer4_1s.txt',
            'premise_question_answer4_2s.txt',
            'premise_question_answer5_1s.txt',
            'premise_question_answer5_2s.txt',
           ]
    for p in px2:
        px.append((p, False))

for p, is_handmade in px:
    print('Parsing {:<40s} '.format(p), end='')
    nb_patterns0 = 0
    with io.open(os.path.join(data_folder, p), "r", encoding="utf-8") as inf:

        loading_state = 'T'

        text = []
        questions = []  # список пар (вопрос, ответ)

        # В автоматически сгенерированных корпусах вопросов-ответов может быть
        # очень много записей, некоторые из которых могут быть не совсем качественные.
        # Поэтому ограничим их число в пользу вручную сформированного корпуса.
        max_samples = 10000000
        filter_answer_complexity = False
        if p != qa_path:
            max_samples = MAX_AUTOGEN_SAMPLES
            filter_answer_complexity = True

        samples_count = 0
        eof = False
        while samples_count <= max_samples and not eof:
            line = inf.readline()
            if len(line) == 0:
                line  = u''
                eof = True
            else:
                line = line.strip()

            if line.startswith(u'T:') or eof:
                if loading_state == 'T':
                    text.append(normalize_qline(line))
                else:
                    # закончился парсинг предыдущего блока из текста (предпосылки),
                    # вопросов и ответов.
                    if len(text) == 1:
                        for premise in text:
                            for question in questions:
                                answer = question[1]
                                if not filter_answer_complexity or len(answer) >= 6 or u' ' in answer:
                                    # Исключаем паттерны с ответами типа "я" и "ты", так как их
                                    # очень много, и для тренировки сеточного генератора ответов
                                    # они дают бесполезную нагрузку.

                                    if answer == 'да' and not is_handmade:
                                        # ограничим кол-во автоматических сэмплов с ответом 'да'
                                        if random.random() < 0.1:
                                            samples_count += 1
                                            add_record(premise, question[0], answer, is_handmade)
                                            nb_patterns0 += 1
                                    else:
                                        add_record(premise, question[0], answer, is_handmade)
                                        samples_count += 1
                                        nb_patterns0 += 1

                    loading_state = 'T'
                    questions = []
                    text = [normalize_qline(line)]

            elif line.startswith(u'Q:'):
                loading_state = 'Q'
                q = normalize_qline(line)
                a = normalize_qline(inf.readline().strip())
                questions.append((q, a))

    print('{:<6d} patterns'.format(nb_patterns0))
# ---------------------------------------------------------------------------------------

# Все паттерны из contradictions.txt интерпретируются с ответом "НЕТ"
with io.open(os.path.join(data_folder, contradictions_path), 'r', encoding='utf-8') as rdr:
    buf = []
    for line in rdr:
        line = line.strip()
        if len(line) == 0:
            if len(buf) == 2:
                premise = normalize_qline(buf[0])
                question = normalize_qline(buf[1])
                add_record(premise, question, u'нет', True)
                buf = []
        else:
            buf.append(line)

# ---------------------------------------------------------------------------------------

print('Total number of samples={}'.format(len(records)))

# сохраним получившийся датасет в TSV

premises = set()
with io.open(os.path.join(tmp_folder, 'premises.txt'), 'w', encoding='utf-8') as wrt:
    for (premise, question, answer, is_handmade) in records:
        if is_handmade:
            if premise not in premises:
                wrt.write(u'{}\n'.format(premise))
                premises.add(premise)

questions = set()
with io.open(os.path.join(tmp_folder, 'questions.txt'), 'w', encoding='utf-8') as wrt:
    for (premise, question, answer, is_handmade) in records:
        if is_handmade:
            if question not in questions:
                wrt.write(u'{}\n'.format(question))
                questions.add(question)


# намного быстрее будет писать строки сразу в tsv файл на диск.
if True:
    with codecs.open(result_path, 'w', 'utf-8') as wrt:
        wrt.write(u'premise\tquestion\tanswer\n')
        for premise, question, answer, is_handmade in records:
            wrt.write(u'{}\t{}\t{}\n'.format(premise, question, answer))
    df = pd.read_csv(result_path, encoding='utf-8', delimiter='\t', quoting=3)
else:
    df = pd.DataFrame(index=None, columns=['premise', 'question', 'answer'])
    for i, (premise, question, answer, is_handmade) in tqdm.tqdm(enumerate(records),
                                                                 total=len(records),
                                                                 desc='Adding rows to dataframe'):
        df.loc[i] = {'premise':premise, 'question':question, 'answer':answer}

print('Random permutation of dataframe...')
df = sklearn.utils.shuffle(df)

print(u'Writing dataframe to "{}"'.format(result_path))
df.to_csv(result_path, sep='\t', encoding='utf-8', header=True, index=False, quoting=csv.QUOTE_NONE)

# Отдельно сохраним датасет для тренировки yes/no классификатора, добавив туда
# сэмплы с несколькими предпосылками.
input_path = os.path.join(data_folder, 'qa_multy.txt')
samples_yesno = []
samples_all = []
with codecs.open(input_path, 'r', 'utf-8') as rdr:
    premises = []
    question = u''
    answer = u''
    prev_line = u''
    for iline, line in enumerate(rdr):
        line = line.strip()
        if len(line) == 0:
            if len(prev_line) != 0:
                if len(answer) ==0 or len(question) == 0:
                    print('Empty answer or question in sample near line #{} in file "{}"'.format(iline+1, input_path))

                if len(premises) in (0, 1, 2):
                    sample = Sample(premises, question, answer)
                    samples_all.append(sample)
                    if sample.answer in [u'да', u'нет']:
                        samples_yesno.append(sample)

                premises = []
                question = u''
                answer = u''
                prev_line = line
        else:
            prev_line = line
            if line.startswith(u'T:'):
                premises.append(normalize_qline(line))
            elif line.startswith(u'Q:'):
                if len(question) != 0:
                    print(u'Second question "{}" detected for premise(s) "{}"'.format(question, premises[0]))
                    exit(1)
                question = normalize_qline(line)
            elif line.startswith(u'A:'):
                assert(len(answer) == 0)
                answer = normalize_qline(line)

# Добавляем сэмплы с одной предпосылкой
for irow, row in df.iterrows():
    premise = row['premise']
    question = row['question']
    answer = row['answer']
    sample = Sample([premise], question, answer)
    samples_all.append(sample)
    if answer in [u'да', u'нет']:
        samples_yesno.append(sample)


nb_yes = 0
nb_no = 0
for sample in samples_yesno:
    if sample.answer == u'да':
        nb_yes += 1
    else:
        nb_no += 1

print('{} samples for yes/no classification: nb_yes={} nb_no={}'.format(len(samples_yesno), nb_yes, nb_no))

print('Writing {}'.format(pqa_yesno_path))
with codecs.open(pqa_yesno_path, 'w', 'utf-8') as wrt:
    for isample, sample in enumerate(samples_yesno):
        if isample > 0:
            wrt.write('\n')
        sample.save(wrt)

print('Writing {}'.format(pqa_all_path))
with codecs.open(pqa_all_path, 'w', 'utf-8') as wrt:
    for isample, sample in enumerate(samples_all):
        if isample > 0:
            wrt.write('\n')
        sample.save(wrt)

print('Statistics for premise-query-answer dataset...')
df = pd.read_csv(result_path, encoding='utf-8', delimiter='\t', quoting=3)
premise_words = set()
question_words = set()
answer_words = set()

for i, row in df.iterrows():
    phrase = row['answer']
    answer_words.update(tokenizer.tokenize(phrase))

    phrase = row['premise']
    premise_words.update(tokenizer.tokenize(phrase))

    phrase = row['question']
    question_words.update(tokenizer.tokenize(phrase))

for sample in samples_yesno:
    for phrase in sample.premises:
        premise_words.update(tokenizer.tokenize(phrase))
    question_words.update(tokenizer.tokenize(sample.question))
    answer_words.update(tokenizer.tokenize(sample.answer))

all_words = answer_words | premise_words | question_words

print('Vocabulary:')
print('premise  ==> {}'.format(len(premise_words)))
print('question ==> {}'.format(len(question_words)))
print('answer   ==> {}'.format(len(answer_words)))
print('total    ==> {}'.format(len(all_words)))

#print('\nyes/no balance:')
#print('# of yes={}'.format(n_yes))
#print('# of no={}'.format(n_no))

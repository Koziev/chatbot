# -*- coding: utf-8 -*-
'''
Парсим файл с текстами, вопросами и ответами, готовим датасет
для тренировки моделей генерации текста ответа из пары предпосылка+вопрос.
(c) by Koziev Ilya inkoziev@gmail.com
'''

from __future__ import division
from __future__ import print_function

import codecs
import os
import tqdm

import pandas as pd
import sklearn.utils

from utils.tokenizer import Tokenizer

tmp_folder = '../tmp'
data_folder = '../data'
paraphrases_path = '../data/paraphrases.txt'
qa_path = '../data/qa.txt'

USE_AUTOGEN = True  # добавлять ли сэмплы из автоматически сгенерированных датасетов


# ---------------------------------------------------------------


def ru_sanitize(s):
    return s.replace(u'ё', u'е')


def normalize_qline( line ):
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


# ---------------------------------------------------------------

tokenizer = Tokenizer()

records = [] # список из (предпосылка, вопрос, ответ, паттерн_создан_вручную)

added_records_set = set() # для предотвращения повторов

def add_record(premise, question, answer, is_handmade):
    s1 = ru_sanitize(premise.strip())
    s2 = ru_sanitize(question.strip())
    s3 = ru_sanitize(answer.strip())

    k = s1+u'|'+s2+u'|'+s3
    if k not in added_records_set:
        added_records_set.add(k)
        records.append((s1, s2, s3, is_handmade))

# ------------------------------------------------------------------------

print('Parsing {:<40s} '.format(paraphrases_path), end='')
nb_paraphrases = 0
lines = []
with codecs.open(paraphrases_path, "r", "utf-8") as inf:
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
                    for question in questions:
                        add_record( premise, question, u'да', False )
                        nb_paraphrases += 1

            lines = []
        else:
            lines.append(line)

print('{:<6d} patterns'.format(nb_paraphrases))

# ------------------------------------------------------------------------

# 1) Используем созданные вручную паттерны из qa_path
# 2) Добавляем автоматически извлеченные паттерны
px = [ (qa_path, True) ]

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
    with codecs.open(os.path.join(data_folder, p), "r", "utf-8") as inf:

        loading_state = 'T'

        text = []
        questions = []  # список пар (вопрос, ответ)

        # В автоматически сгенерированных корпусах вопросов-ответов может быть
        # очень много записей, некоторые их которых могут быть не совсем качественные.
        # Поэтому ограничим их число в пользу вручную сформированного корпуса.
        max_samples = 10000000
        if p != qa_path:
            max_samples = 5000

        samples_count = 0
        while samples_count <= max_samples:
            line = inf.readline()
            if len(line) == 0:
                break

            line = line.strip()

            if line.startswith(u'T:'):
                if loading_state == 'T':
                    text.append(normalize_qline(line))
                else:
                    # закончился парсинг предыдущего блока из текста (предпосылки),
                    # вопросов и ответов.
                    if len(text) == 1:
                        for premise in text:
                            for question in questions:
                                add_record(premise, question[0], question[1], is_handmade)
                                nb_patterns0 += 1
                                samples_count += 1

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

print('Total number of samples={}'.format(len(records)))

# сохраним получившийся датасет в TSV

premises = set()
with codecs.open(os.path.join(tmp_folder, 'premises.txt'), 'w', 'utf-8') as wrt:
    for (premise, question, answer, is_handmade) in records:
        if is_handmade:
            if premise not in premises:
                wrt.write(u'{}\n'.format(premise))
                premises.add(premise)

questions = set()
with codecs.open(os.path.join(tmp_folder, 'questions.txt'), 'w', 'utf-8') as wrt:
    for (premise, question, answer, is_handmade) in records:
        if is_handmade:
            if question not in questions:
                wrt.write(u'{}\n'.format(question))
                questions.add(question)

result_path = os.path.join(data_folder, 'premise_question_answer.csv')

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

print(u'Writing dataframe to {}'.format(result_path))
df.to_csv(result_path, sep='\t', encoding='utf-8', header=True, index=False)

print('Statistics...')
df = pd.read_csv(result_path, encoding='utf-8', delimiter='\t', quoting=3)
premise_words = set()
question_words = set()
answer_words = set()

n_yes = 0
n_no = 0

for i, row in df.iterrows():
    phrase = row['answer']
    answer_words.update(tokenizer.tokenize(phrase))
    if phrase.lower() == u'да':
        n_yes += 1
    elif phrase.lower() == u'нет':
        n_no += 1

    phrase = row['premise']
    premise_words.update(tokenizer.tokenize(phrase))

    phrase = row['question']
    question_words.update(tokenizer.tokenize(phrase))

all_words = answer_words | premise_words | question_words

print('Vocabulary:')
print('premise  ==> {}'.format(len(premise_words)))
print('question ==> {}'.format(len(question_words)))
print('answer   ==> {}'.format(len(answer_words)))
print('total    ==> {}'.format(len(all_words)))

print('\nyes/no balance:')
print('# of yes={}'.format( n_yes ))
print('# of no={}'.format( n_no ))

# -*- coding: utf-8 -*-
'''
Парсим файл с текстами, вопросами и ответами, готовим датасет
для тренировки моделей генерации текста ответа из пары предпосылка+вопрос.
(c) by Koziev Ilya inkoziev@gmail.com
'''

from __future__ import print_function
from __future__ import division  # for python2 compatability

import codecs
import itertools
import random
import os
import pandas as pd

from Tokenizer import Tokenizer


tmp_folder = '../tmp'
data_folder = '../data'
paraphrases_path = '../data/paraphrases.txt'
qa_path = '../data/qa.txt'

# ---------------------------------------------------------------

def normalize_qline( line ):
    line = line.replace(u'(+)', u'')
    line = line.replace(u'(-)', u'')
    line = line.replace(u'T:', u'')
    line = line.replace(u'Q:', u'')
    line = line.replace(u'A:', u'')
    line = line.replace(u'\t', u' ')
    line = line.replace( '.', ' ' ).replace( ',', ' ' ).replace( '?', ' ' ).replace( '!', ' ' ).replace( '-', ' ' )
    line = line.replace( '  ', ' ' ).strip().lower()
    return line


# ---------------------------------------------------------------

tokenizer = Tokenizer()

# ------------------------------------------------------------------------

records = [] # список из (предпосылка, вопрос и ответ)

added_records_set = set() # для предотвращения повторов

def add_record( premise, question, answer ):

    s1 = premise.strip()
    s2 = question.strip()
    s3 = answer.strip()

    k = s1+u'|'+s2+u'|'+s3
    if k not in added_records_set:
        added_records_set.add(k)
        records.append((s1, s2, s3))

# ------------------------------------------------------------------------

print('Parsing {}'.format(paraphrases_path))
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
                        if line[-1]==u'?':
                            questions.append(s)
                        else:
                            premises.append(s)
                            questions.append(s)

                for premise in premises:
                    for question in questions:
                        add_record( premise, question, u'да' )

            lines = []
        else:
            lines.append(line)


# ------------------------------------------------------------------------

# 1) Используем созданные вручную паттерны из qa_path
# 2) Добавляем автоматически извлеченные праттерны
px = [ qa_path,
       'current_time_pqa.txt',
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


for p in px:
    print('Parsing {}'.format(p))
    nb_patterns0 = 0
    with codecs.open(os.path.join(data_folder, p), "r", "utf-8") as inf:

        loading_state = 'T'

        text = []
        questions = [] # список (вопрос, ответ)

        while True:
            line = inf.readline()
            if len(line)==0:
                break

            line = line.strip()

            if line.startswith(u'T:'):
                if loading_state=='T':
                    text.append(normalize_qline(line))
                else:
                    # закончился парсинг предыдущего блока из текста (предпосылки),
                    # вопросов и ответов.
                    if len(text)==1:
                        for premise in text:
                            for question in questions:
                                add_record( premise, question[0], question[1] )
                                nb_patterns0 += 1

                    loading_state = 'T'
                    questions = []
                    text = [normalize_qline(line)]

            elif line.startswith(u'Q:'):
                loading_state = 'Q'
                q = normalize_qline(line)
                a = normalize_qline(inf.readline().strip())
                questions.append((q, a))

    print('{} patterns extracted'.format(nb_patterns0))
# ---------------------------------------------------------------------------------------

print('Total number of samples={}'.format(len(records)))

result_path = os.path.join(data_folder,'premise_question_answer.csv')

# сохраним получившийся датасет в CSV
with codecs.open(result_path, 'w', 'utf-8') as wrt:
    wrt.write(u'premise\tquestion\tanswer\n')
    for (premise, question, answer) in records:
        wrt.write(u'{}\t{}\t{}\n'.format(premise, question, answer))

# ----------------------------------------------------------------------------------------

# Для проверки загрузим сохраненный датасет, соберем некоторую статистику
df = pd.read_csv(result_path, encoding='utf-8', delimiter='\t', quoting=3)

premise_words = set()
question_words = set()
answer_words = set()

n_yes = 0
n_no = 0

for i,row in df.iterrows():
    phrase = row['answer']
    answer_words.update(tokenizer.tokenize(phrase))
    if phrase.lower()==u'да':
        n_yes += 1
    elif phrase.lower()==u'нет':
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

# -*- coding: utf-8 -*-
'''
Подготовка списка слов для тренировки модели
встраиваний слова wordchar2vector.

(c) by Koziev Ilya inkoziev@gmail.com для чат-бота https://github.com/Koziev/chatbot
'''

from __future__ import print_function

import codecs
import itertools
import os
from sys import platform
import pandas as pd

from utils.tokenizer import Tokenizer

result_path = '../tmp/known_words.txt' # путь к файлу, где будет сохранен список слов

n_misspelling_per_word = 0 # кол-во добавляемых вариантов с опечатками на одно исходное слово


# Из этого текстового файла возьмем слова, на которых будем тренировать модель встраивания.
if platform == "win32":
    corpus_path = r'f:\Corpus\word2vector\ru\SENTx.corpus.w2v.txt'
else:
    corpus_path = os.path.expanduser('~/Corpus/word2vector/ru/SENTx.corpus.w2v.txt')

paraphrases_path = '../data/premise_question_relevancy.csv'
pqa_path = '../data/premise_question_answer.csv'
eval_path = '../data/evaluate_relevancy.txt'

# ---------------------------------------------------------------

known_words = set()

# Берем слова из большого текстового файла, на котором тренируется w2v модели.
print('Parsing {}'.format(corpus_path))
with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
    line_count = 0
    for line0 in rdr:
        line = line0.strip()
        words = line.split(u' ')
        known_words.update(words)
        line_count += 1
        if line_count>1000000:
            break

# Добавим слова из основного тренировочного датасета
print('Parsing {}'.format(paraphrases_path))
df = pd.read_csv(paraphrases_path, encoding='utf-8', delimiter='\t', quoting=3)
tokenizer = Tokenizer()
for phrase in itertools.chain(df['premise'].values, df['question'].values):
    words = tokenizer.tokenize(phrase.lower())
    known_words.update(words)

print('Parsing {}'.format(pqa_path))
df = pd.read_csv(pqa_path, encoding='utf-8', delimiter='\t', quoting=3)
for phrase in itertools.chain( df['premise'].values, df['question'].values, df['answer'].values ):
    words = tokenizer.tokenize(phrase)
    known_words.update(words)

# Добавим слова, которые употребляются в датасете для оценки
print('Parsing {}'.format(eval_path))
with codecs.open(eval_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        phrase = line.replace(u'T:', u'').replace(u'Q:', u'').strip()
        words = tokenizer.tokenize(phrase)
        known_words.update(words)

print('There are {} known words'.format(len(known_words)))

stop_words = {u'_num_'}

with codecs.open(result_path, 'w', 'utf-8') as wrt:
    for word in sorted(known_words):
        if word not in stop_words and not word.startswith(u' '):
            wrt.write(u'{}\n'.format(word))


# -*- coding: utf-8 -*-
"""
Генерация нерелевантных предпосылок для вопросов в датасете.
Используется в ходе полуавтоматического наполнения датасета "nonrelevant_premise_questions.txt"
Список предпосылок загружается из большого текстового корпуса. Строится
обратный индекс. Затем для каждого вопроса в qa.txt подьирается некоторое количество
похожих предпосылок, которые далее будут вручную модерироваться.
"""

from __future__ import division
from __future__ import print_function

import codecs
import itertools
import json
import functools
import os
import queue
import sys
import argparse
import random
import math
import pandas as pd
import numpy as np
import re
import tqdm
import concurrent.futures
import multiprocessing

from utils.tokenizer import Tokenizer
from utils.segmenter import Segmenter
from preparation.corpus_searcher import CorpusSearcher


data_folder = '../../data'
tmp_folder = '../../tmp'


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


def plain(s):
    return u' '.join(word for word in tokenizer.tokenize(s) if len(word) > 0)


rx = re.compile('[():a-zA-Z*{}_]')


def is_good_premise(phrase):
    if rx.search(phrase) is None and not phrase[0] in u'-–,».':
        words = tokenizer.tokenize(phrase)
        if len(phrase) < 80 and 8 > len(words) > 3:
            return True

    return False


df = pd.read_csv(os.path.join(data_folder, 'premise_question_relevancy.csv'), encoding='utf-8', delimiter='\t', quoting=3)
added_pq = set((premise+'|'+question) for premise, question in zip(df['premise'].values, df['question'].values))

segmenter = Segmenter()
tokenizer = Tokenizer()
tokenizer.load()

random_facts = CorpusSearcher()

# Прочитаем список случайных фактов, чтобы потом генерировать отрицательные паттерны
corpus_path = os.path.expanduser('~/Corpus/Raw/ru/text_blocks.txt')
n = 0
print(u'Loading samples from {}'.format(corpus_path))
with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        phrases = segmenter.split(line)
        for phrase in phrases:
            if phrase[-1] == '.':
                phrase = phrase.strip().replace('--', '-')
                if phrase.count('"') == 1:
                    phrase = phrase.replace('"', '')

                if is_good_premise(phrase):
                    random_facts.add_phrase(phrase)
                    n += 1

        if n > 5000000:
            break

print('{} random facts in set'.format(len(random_facts)))


# Для этих вопросов негативные сэмплы уже подобраны
processed_questions = set()
with codecs.open(os.path.join(data_folder, 'nonrelevant_premise_questions.txt'), 'r', 'utf-8') as rdr:
    for line in rdr:
        if '|' in line:
            question = plain(line.strip().split('|')[1])
            processed_questions.add(question)


# Собираем список вопросов, для которых будем подбирать похожие нерелевантные предпосылки
questions = set()
with codecs.open(os.path.join(data_folder, 'qa.txt'), 'r', 'utf-8') as rdr:
#with codecs.open(os.path.join(data_folder, 'premise_question_answer4.txt'), 'r', 'utf-8') as rdr:
#with codecs.open(os.path.join(data_folder, 'simple_questions.txt'), 'r', 'utf-8') as rdr:
    for line in rdr:
        if line.startswith('Q:'):
            q = line.replace('Q:', '').strip()
            if plain(q) not in processed_questions:
                questions.add(q)
questions = list(questions)
questions = list(np.random.permutation(questions))[:100]  # <--DEBUG!!!
print('There are {} questions to process'.format(len(questions)))


def collect_similar_premises(question, corpus_searcher, added_pq):
    results = []
    premises = corpus_searcher.find_similar(question, 15)
    for premise in premises:
        k = plain(premise) + '|' + plain(question)
        if k not in added_pq:
            results.append((premise, question))
    return results


if True:
    # Однопоточный вариант
    with codecs.open(os.path.join(tmp_folder, 'raw_nonrelevant_premises.txt'), 'w', 'utf-8') as wrt:
        for question in tqdm.tqdm(questions, total=len(questions), desc='Searching similar premises'):
            premises = collect_similar_premises(question, random_facts, added_pq)
            for premise, question in premises:
                wrt.write(u'{:<60s}  |  {}\n'.format(premise, question))

            wrt.write('\n\n')
            wrt.flush()
else:
    result_queue = multiprocessing.Queue()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm.tqdm(executor.map(functools.partial(collect_similar_premises, corpus_searcher=random_facts, added_pq=added_pq), questions),
                                total=len(questions),
                                desc='Searching'):
            result_queue.put(result)

    print('Writing results...')
    with codecs.open(os.path.join(tmp_folder, 'raw_nonrelevant_premises.txt'), 'w', 'utf-8') as wrt:
        try:
            while True:
                result = result_queue.get(False)
                for premise, question in result:
                    k = plain(premise) + '|' + plain(question)
                    if k not in added_pq:
                        wrt.write(u'{:<60s}  |  {}\n'.format(premise, question))
                        added_pq.add(k)

                wrt.write('\n\n')
                wrt.flush()
        except queue.Empty as e:
            print('Queue of results is empty')

print('All done.')




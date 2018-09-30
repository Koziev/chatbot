# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки моделей, определяющих СИНОНИМИЮ - семантическую
эквивалентности двух фраз, включая позиционные и синтаксические перефразировки, лексические и фразовые
синонимы. В отличие от модели для РЕЛЕВАНТНОСТИ предпосылки и вопроса, в этой
модели предполагается, что объем информации в обеих фразах примерно одинаков,
то есть "кошка спит" и "черная кошка сладко спит" не считаются полными синонимами.

Для проекта чатбота https://github.com/Koziev/chatbot
"""

from __future__ import division
from __future__ import print_function

import codecs
import itertools
import operator
import os
import sys
import argparse
import random
import collections
import logging
import tqdm
import numpy as np

from utils.tokenizer import Tokenizer
import utils.logging_helpers


nb_neg_per_posit = 1

random.seed(123456789)
np.random.seed(123456789)


class Sample:
    def __init__(self, phrase1, phrase2, y):
        assert(phrase1 > 0)
        assert(phrase2 > 0)
        assert(y in [0, 1])
        self.phrase1 = phrase1
        self.phrase2 = phrase2
        self.y = y


class NegSamples:
    def __init__(self):
        self.neg_samples = []
        self.neg_samples_set = set()

    @staticmethod
    def key(s1, s2):
        return s1 + u'|' + s2

    def add(self, phrase1, phrase2):
        k1 = NegSamples.key(phrase1, phrase2)
        if k1 not in self.neg_samples_set:
            k2 = NegSamples.key(phrase2, phrase1)
            if k2 not in self.neg_samples_set:
                self.neg_samples_set.add(k1)
                self.neg_samples_set.add(k2)
                self.neg_samples.append(Sample(phrase1, phrase2, 0))

    def get_samples(self):
        return self.neg_samples


def ngrams(s, n):
    return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1&shingles2))/float(1e-8+len(shingles1|shingles2))


def select_most_similar(phrase1, phrases2, topn):
    sims = [(phrase2, jaccard(phrase1, phrase2, 3)) for phrase2 in phrases2]
    sims = sorted(sims, key=lambda z: -z[1])
    return list(map(operator.itemgetter(0), sims[:topn]))


class PhraseCleaner:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def process(self, phrase):
        return u' '.join(self.tokenizer.tokenize(phrase))



tmp_folder = '../tmp'
data_folder = '../data'
input_path = '../data/paraphrases.txt'
output_filepath = '../data/synonymy_dataset.csv'

# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'prepare_synonymy_dataset.log'))

logging.info('Start')

fcleaner = PhraseCleaner()

# Для генерации негативных сэмплов нам надо исключать
# вероятность попадания перефразировок в качестве негативных
# примеров для групп, содержащих более 2 вариантов. Поэтому
# каждую фразу пометим номеров ее исходной группы.
phrase2group = dict()

# Грузим датасеты с перефразировками
samples = []  # список из экземпляров Sample
igroup = 0
group = []
with codecs.open(input_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        phrase = line.strip()
        if len(phrase) == 0:
            if len(group) > 1:
                igroup += 1
                n = len(group)
                for i1 in range(n):
                    phrase1 = group[i1]
                    if phrase1.startswith(u'(-)'):
                        continue

                    if phrase1.startswith(u'(+)'):
                        phrase1 = phrase1.replace(u'(+)', u'').strip()

                    phrase1 = fcleaner.process(phrase1)

                    phrase2group[phrase1] = igroup
                    for i2 in range(i1 + 1, n):
                        y = 1
                        phrase2 = group[i2]

                        if phrase2.startswith(u'(+)'):
                            phrase2 = phrase2.replace(u'(+)', u'').strip()
                        elif phrase2.startswith(u'(-)'):
                            phrase2 = phrase2.replace(u'(-)', u'').strip()
                            y = 0
                        phrase2 = fcleaner.process(phrase2)

                        phrase2group[phrase2] = igroup
                        samples.append(Sample(phrase1, phrase2, y))
                        if phrase1 != phrase2:
                            # меняем местами сравниваемые фразы
                            samples.append(Sample(phrase2, phrase1, y))

                group = []
        else:
            group.append(phrase)

logging.info('{} pairs loaded from "{}"'.format(len(samples), input_path))

# Из датасета для "антонимов" берем обязательные негативные примеры.
group = []
nb_antonyms = 0
with codecs.open(os.path.join(data_folder, 'contradictions.txt'), 'r', 'utf-8') as rdr:
    for line in rdr:
        phrase = line.strip()
        if len(phrase) == 0:
            if len(group) > 1:
                igroup += 1
                n = len(group)
                for i1 in range(n):
                    phrase1 = group[i1]
                    phrase1 = fcleaner.process(phrase1)
                    phrase2group[phrase1] = igroup
                    for i2 in range(i1 + 1, n):
                        phrase2 = group[i2]
                        phrase2 = fcleaner.process(phrase2)
                        phrase2group[phrase2] = igroup
                        samples.append(Sample(phrase1, phrase2, 0))
                        nb_antonyms += 1

                        if phrase1 != phrase2:
                            # меняем местами сравниваемые фразы
                            samples.append(Sample(phrase2, phrase1, 0))
                            nb_antonyms += 1

                group = []
        else:
            group.append(phrase)

logging.info('{} antonyms loaded from \"contradictions.txt\"'.format(nb_antonyms))

# добавим некоторое кол-во идентичных пар.
all_phrases = set()
for sample in samples:
    all_phrases.add(sample.phrase1)
    all_phrases.add(sample.phrase2)

duplicates = []
for phrase in all_phrases:
    duplicates.append(Sample(phrase, phrase, 1))

duplicates = np.random.permutation(duplicates)[:len(samples)//10]
samples.extend(duplicates)

# дубликаты
duplicates = []
for p in ['SENT4.duplicates.txt', 'SENT5.duplicates.txt', 'SENT6.duplicates.txt']:
    group = []
    with codecs.open(os.path.join(data_folder, p), 'r', 'utf-8') as rdr:
        for line in rdr:
            phrase = line.strip()
            if len(phrase) == 0:
                if len(group) > 1:
                    igroup += 1
                    n = len(group)
                    for i1 in range(n):
                        phrase2group[group[i1]] = igroup
                        for i2 in range(i1+1, n):
                            phrase2group[group[i2]] = igroup
                            duplicates.append(Sample(group[i1], group[i2], 1))
                    group = []
            else:
                group.append(fcleaner.process(phrase))

# оставим кол-во дубликатов, сопоставимое с другими перефразировками
duplicates = np.random.permutation(duplicates)[:len(samples)//2]
logging.info('{} duplicates with permutations loaded from \"SENT*.duplicates.txt\"'.format(len(duplicates)))
samples.extend(duplicates)

all_phrases = set()
for sample in samples:
    if sample.y == 1:
        all_phrases.add(sample.phrase1)
        all_phrases.add(sample.phrase2)

all_phrases = list(all_phrases)

# Для быстрого поиска похожих фраз создадим обратные индексы.
word2phrases = dict()
for phrase in all_phrases:
    words = phrase.split(u' ')
    for word in words:
        if word not in word2phrases:
            word2phrases[word] = [phrase]
        else:
            word2phrases[word].append(phrase)

neg_samples = NegSamples()

# Добавим негативные сэмплы, похожие на фразы в позитивных.
# Это должно обучить модель не полагаться просто на наличие одинаковых слов.
for phrase1 in tqdm.tqdm(all_phrases, total=len(all_phrases), desc='Adding similar negatives'):
    similar_phrases2 = collections.Counter()
    group1 = phrase2group[phrase1]
    for word1 in phrase1.split(u' '):
        if word1 in word2phrases:
            for phrase2 in word2phrases[word1]:
                if phrase2group[phrase2] != group1:
                    similar_phrases2[phrase2] += 1

    # наиболее похожие фразы по числу одинаковых слов
    phrases2_a = similar_phrases2.most_common(10)
    phrases2_a = list(map(operator.itemgetter(0), phrases2_a))

    # Среди similar_phrases2 оставим наиболее похожие на первую фразу,
    # используя коэф-т Жаккара как простую меру сходства
    phrases2_b = select_most_similar(phrase1, similar_phrases2, 10)

    phrases2 = set(itertools.chain(phrases2_a, phrases2_b))

    for phrase2 in phrases2:
        neg_samples.add(phrase1, phrase2)

# Добавляем рандомные негативные сэмплы
for sample in samples:
    igroup1 = phrase2group[sample.phrase1]
    n_neg = 0
    while n_neg < 10:
        neg_phrase = random.choice(all_phrases)
        if neg_phrase not in phrase2group:
            pass

        if phrase2group[neg_phrase] != igroup1:
            neg_samples.add(sample.phrase1, neg_phrase)
            neg_samples.add(sample.phrase2, neg_phrase)
            n_neg += 2

# Сколько позитивных сэмплов
nb_1 = sum(sample.y == 1 for sample in samples)

# ограничим кол-во негативных сэмплов
neg_samples = neg_samples.get_samples()
neg_samples = np.random.permutation(neg_samples)[:nb_1*nb_neg_per_posit]
logging.info('{} negative samples added'.format(len(neg_samples)))
samples.extend(neg_samples)
samples = np.random.permutation(samples)

nb_0 = sum(sample.y == 0 for sample in samples)
logging.info('nb_0={} nb_1={}'.format(nb_0, nb_1))

max_wordseq_len = 0
for sample in samples:
    for phrase in [sample.phrase1, sample.phrase2]:
        words = phrase.split(u' ')
        max_wordseq_len = max(max_wordseq_len, len(words))

logging.info('max_wordseq_len={}'.format(max_wordseq_len))

# сохраним получившийся датасет в CSV
logging.info(u'Storing result dataset to "{}"'.format(output_filepath))
with codecs.open(output_filepath, 'w', 'utf-8') as wrt:
    # Заголовки делаем как у датасета relevancy моделей, чтобы их можно было использовать без переделок.
    wrt.write(u'premise\tquestion\trelevance\tweight\n')
    for sample in samples:
        wrt.write(u'{}\t{}\t{}\t1\n'.format(sample.phrase1, sample.phrase2, sample.y))

logging.info('Finish')

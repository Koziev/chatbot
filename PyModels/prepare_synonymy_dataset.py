# -*- coding: utf-8 -*-
"""
Подготовка датасетов для тренировки моделей, определяющих СИНОНИМИЮ - семантическую
эквивалентность двух фраз, включая позиционные и синтаксические перефразировки,
лексические и фразовые синонимы. В отличие от модели для РЕЛЕВАНТНОСТИ предпосылки
и вопроса (см. nn_relevancy.py и lgb_relevancy.py), в этой модели предполагается,
что объем информации в обеих фразах примерно одинаков, то есть "кошка спит" и
"черная кошка сладко спит" не считаются полными синонимами.

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


# Кол-во негативных сэмплов, автоматически подбираемых на один позитивный пример.
nb_neg_per_posit = 1

ADD_SIMILAR_NEGATIVES = False  # негативные вопросы подбирать по похожести к предпосылке (либо чисто рандомные)

# Путь к файлу с отобранными вручную синонимичными фразами.
input_path = '../data/paraphrases.txt'

# Путь к создаваемому датасету для модели детектора, использующего пары
output_filepath = '../data/synonymy_dataset.csv'

# Путь к создаваемому датасету для модели детектора на базе triplet loss
output_filepath3 = '../data/synonymy_dataset3.csv'

tmp_folder = '../tmp'
data_folder = '../data'


random.seed(123456789)
np.random.seed(123456789)


class Sample:
    def __init__(self, phrase1, phrase2, y):
        assert(len(phrase1) > 0)
        assert(len(phrase2) > 0)
        assert(y in (0, 1))
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




class Sample2:
    def __init__(self, anchor, positive):
        assert(len(anchor) > 0)
        assert(len(positive) > 0)
        self.anchor = anchor
        self.positive = positive


class Sample3:
    def __init__(self, anchor, positive, negative):
        assert(len(anchor) > 0)
        assert(len(positive) > 0)
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

    def key(self):
        return self.anchor + u'|' + self.positive + u'|' + self.negative




def ngrams(s, n):
    return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))


BEG_CHAR = '\b'
END_CHAR = '\n'


def jaccard(s1, s2, shingle_len):
    """
    Вычисляется коэффициент Жаккара для похожести двух строк s1 и s2, которые
    нарезаются на шинглы длиной по shingle_len символов.
    Возвращается действительное число в диапазоне от 0.0 до 1.0 
    """
    shingles1 = ngrams(BEG_CHAR + s1.lower() + END_CHAR, shingle_len)
    shingles2 = ngrams(BEG_CHAR + s2.lower() + END_CHAR, shingle_len)
    return float(len(shingles1 & shingles2)) / float(1e-8 + len(shingles1 | shingles2))


def select_most_similar(phrase1, phrases2, topn):
    """
    Для строки phrase1 в списке строк phrases2 отбираются и возвращаются
    topn самых похожих.
    """
    sims = [(phrase2, jaccard(phrase1, phrase2, 3)) for phrase2 in phrases2]
    sims = sorted(sims, key=lambda z: -z[1])
    return list(map(operator.itemgetter(0), sims[:topn]))


class PhraseCleaner:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def process(self, phrase):
        return u' '.join(self.tokenizer.tokenize(phrase))


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

logging.info('[[[OK]]] {} pairs (positive and negative ones) have been loaded from "{}"'.format(len(samples), input_path))

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

logging.info('{} antonyms loaded from "contradictions.txt"'.format(nb_antonyms))

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
if ADD_SIMILAR_NEGATIVES:
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
neg_samples = np.random.permutation(neg_samples)[:nb_1 * nb_neg_per_posit]
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

# --------------------------------------------------------------------------
# Теперь готовим датасет для модели детектора перефразировок с triplet loss
# Тут нам надо готовить триплеты (anchor, positive, negative)
logging.info('Start building dataset for triplet loss model of synonymy')

# nb_neg_per_posit = 2
samples3 = []  # финальный список из экземпляров Sample, содержащих тройки (anchor, positive, negative)
samples2 = []  # вспомогательный список из экземпляров Sample2, содержащих пары (anchor, positive)

fcleaner = PhraseCleaner()

# Из этого файла берем негативные фразы для троек.
phrase2contradict = dict()
with codecs.open(os.path.join(data_folder, 'contradictions.txt'), 'r', 'utf-8') as rdr:
    group = []
    for line in rdr:
        phrase = line.strip()
        if len(phrase) == 0:
            if len(group) == 2:
                phrase1 = fcleaner.process(group[0])
                phrase2 = fcleaner.process(group[1])

                if phrase1 not in phrase2contradict:
                    phrase2contradict[phrase1] = [phrase2]
                else:
                    phrase2contradict[phrase1].append(phrase2)

                if phrase2 not in phrase2contradict:
                    phrase2contradict[phrase2] = [phrase1]
                else:
                    phrase2contradict[phrase2].append(phrase1)

                group = []
        else:
            group.append(phrase)



# Для генерации негативных сэмплов нам надо исключать
# вероятность попадания перефразировок в качестве негативных
# примеров для групп, содержащих более 2 вариантов. Поэтому
# каждую фразу пометим номером ее исходной группы.
phrase2group = dict()

# Грузим датасеты с перефразировками
igroup = 0
group = []
nb_paraphrases1 = 0
with codecs.open('../data/paraphrases.txt', 'r', 'utf-8') as rdr:
    for line in rdr:
        phrase = line.strip()
        if len(phrase) == 0:
            if len(group) > 1:
                igroup += 1
                n = len(group)

                pos_phrases = []
                neg_phrases = []

                for i1 in range(n):
                    phrase1 = group[i1]
                    sim = 0
                    if phrase1.startswith(u'(-)'):
                        phrase1 = phrase1.replace(u'(-)', u'').strip()
                        sim = -1

                    if phrase1.startswith(u'(+)'):
                        phrase1 = phrase1.replace(u'(+)', u'').strip()
                        sim = 1

                    phrase1 = fcleaner.process(phrase1)
                    phrase2group[phrase1] = igroup
                    if sim == -1:
                        neg_phrases.append(phrase1)
                    else:
                        pos_phrases.append(phrase1)

                # сочетания положительных фраз
                for i1 in range(len(pos_phrases) - 1):
                    phrase1 = pos_phrases[i1]
                    for i2 in range(1, len(pos_phrases)):
                        phrase2 = pos_phrases[i2]
                        if phrase1 != phrase2:  # исключаем сэмплы, в которых постулируется похожесть одинаковых предложений
                            nb_paraphrases1 += 1
                            samples2.append(Sample2(phrase1, phrase2))

                            # если есть негативные фразы, то сразу можем генерировать
                            # финальные триплеты.
                            for negative in neg_phrases:
                                samples3.append(Sample3(phrase1, phrase2, negative))

                            if phrase1 in phrase2contradict:
                                for negative in phrase2contradict[phrase1]:
                                    samples3.append(Sample3(phrase1, phrase2, negative))

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

                        if y == 1:
                            phrase2 = fcleaner.process(phrase2)

                            phrase2group[phrase2] = igroup

                            samples2.append(Sample2(phrase1, phrase2))
                            if phrase1 != phrase2:
                                # меняем местами сравниваемые фразы
                                samples2.append(Sample2(phrase2, phrase1))

                group = []
        else:
            group.append(phrase)

logging.info('[[[OK]]] {} positive pairs have been loaded from "{}"'.format(nb_paraphrases1, input_path))

# добавим некоторое кол-во идентичных пар.
all_phrases = set()
for sample in samples2:
    all_phrases.add(sample.anchor)
    all_phrases.add(sample.positive)

duplicates = []
for phrase in all_phrases:
    duplicates.append(Sample2(phrase, phrase))

duplicates = np.random.permutation(duplicates)[:len(samples2) // 10]
samples2.extend(duplicates)

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
                        for i2 in range(i1 + 1, n):
                            phrase2group[group[i2]] = igroup
                            duplicates.append(Sample2(group[i1], group[i2]))
                    group = []
            else:
                group.append(fcleaner.process(phrase))

# оставим кол-во дубликатов, сопоставимое с другими перефразировками
duplicates = np.random.permutation(duplicates)[:len(samples2) // 2]
logging.info('{} duplicates with permutations loaded from \"SENT*.duplicates.txt\"'.format(len(duplicates)))
samples2.extend(duplicates)

all_phrases = set()
for sample in samples2:
    all_phrases.add(sample.anchor)
    all_phrases.add(sample.positive)

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

# Идем по накопленным сэмплам и добавляем для каждого негативную фразу
for sample in tqdm.tqdm(samples2, total=len(samples2), desc='Adding negative phrases'):
    phrase1 = sample.anchor
    group1 = phrase2group[phrase1]
    if ADD_SIMILAR_NEGATIVES:
        similar_phrases2 = collections.Counter()
        for word1 in phrase1.split(u' '):
            if word1 in word2phrases:
                for phrase2 in word2phrases[word1]:
                    if phrase2group[phrase2] != group1:
                        similar_phrases2[phrase2] += 1

        # наиболее похожие фразы по числу одинаковых слов
        phrases2_a = similar_phrases2.most_common(2)
        phrases2_a = list(map(operator.itemgetter(0), phrases2_a))

        # Среди similar_phrases2 оставим наиболее похожие на первую фразу,
        # используя коэф-т Жаккара как простую меру сходства
        # phrases2_b = select_most_similar(phrase1, similar_phrases2, 2)
        # phrases2 = set(itertools.chain(phrases2_a, phrases2_b))
        phrases2 = phrases2_a
    else:
        phrases2 = []

    if len(phrases2) < nb_neg_per_posit:
        # выберем просто рандомную строку
        for _ in range(nb_neg_per_posit):
            phrase2 = random.choice(all_phrases)
            if phrase2group[phrase2] != group1:
                phrases2.append(phrase2)
                break

    # берем несколько фраз, делаем соответствующее кол-во полных сэмплов
    for negative in phrases2[:nb_neg_per_posit]:
        sample3 = Sample3(sample.anchor, sample.positive, negative)
        samples3.append(sample3)

# Отфильтруем повторы, вдруг они появятся.
samples3 = dict((s.key(), s) for s in samples3).values()

# сохраним получившийся датасет в CSV
logging.info(u'Storing {} triplets to dataset "{}"'.format(len(samples), output_filepath3))
with codecs.open(output_filepath3, 'w', 'utf-8') as wrt:
    # Заголовки делаем как у датасета relevancy моделей, чтобы их можно было использовать без переделок.
    wrt.write(u'anchor\tpositive\tnegative\n')
    for sample in samples3:
        wrt.write(u'{}\t{}\t{}\n'.format(sample.anchor, sample.positive, sample.negative))

logging.info('Finish')

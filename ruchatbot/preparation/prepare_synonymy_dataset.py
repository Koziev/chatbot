# -*- coding: utf-8 -*-
"""
Подготовка датасетов для тренировки моделей, определяющих СИНОНИМИЧНОСТЬ - семантическую
эквивалентность двух фраз, включая позиционные и синтаксические перефразировки,
лексические и фразовые синонимы. В отличие от модели для РЕЛЕВАНТНОСТИ предпосылки
и вопроса (см. nn_relevancy.py и lgb_relevancy.py), в этой модели предполагается,
что объем информации в обеих фразах примерно одинаков, то есть "кошка спит" и
"черная кошка сладко спит" не считаются полными синонимами.

Для проекта чатбота https://github.com/Koziev/chatbot

01.04.2019 Добавляем негативные примеры из файла nonrelevant_premise_questions.txt
08-11-2019 Использование networkx для определения синонимичности фраз, описанных в разных блоках
"""

from __future__ import division
from __future__ import print_function

import io
import codecs
import itertools
import operator
import os
import re
import sys
import argparse
import random
import collections
import logging
import tqdm
import numpy as np
import networkx as nx

from ruchatbot.utils.tokenizer import Tokenizer
import ruchatbot.utils.logging_helpers


# Кол-во негативных сэмплов, автоматически подбираемых на один позитивный пример.
nb_neg_per_posit = 1

ADD_SIMILAR_NEGATIVES = False  # негативные вопросы подбирать по похожести к предпосылке (либо чисто рандомные)

tmp_folder = '../../tmp'
data_folder = '../../data'

# Путь к файлу с отобранными вручную синонимичными фразами.
input_path = '../../data/paraphrases.txt'

# Путь к создаваемому датасету для модели детектора, использующего пары
output_filepath = '../../data/synonymy_dataset.csv'

# Путь к создаваемому датасету для модели детектора на базе triplet loss
output_filepath3 = '../../data/synonymy_dataset3.csv'


random.seed(123456789)
np.random.seed(123456789)


def is_int(s):
    return re.match(r'^\d+$', s)


class Sample:
    def __init__(self, phrase1, phrase2, y):
        if len(phrase1) == 0 or len(phrase2) == 0:
            logging.error(u'Empty phrase: phrase1={} phrase2={}'.format(phrase1, phrase2))
            raise RuntimeError()

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
        self.anchor = anchor.strip()
        self.positive = positive.strip()
        self.negative = negative.strip()

    def key(self):
        return self.anchor + u'|' + self.positive + u'|' + self.negative


def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


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
        self.tokenizer.load()

    def process(self, phrase):
        return u' '.join(self.tokenizer.tokenize(phrase))


class Samples:
    def __init__(self):
        self.samples = []
        self.sample2source = dict()
        self.pair2label = dict()

    def append(self, sample, source):
        if sample.y == 0 and sample.phrase1 == sample.phrase2:
            logging.error('Adding identity pair with label=0: phrase1="%s", phrase2="%s", source=%d', sample.phrase1, sample.phrase2, source)
        else:
            self.samples.append(sample)
            p = (sample.phrase1, sample.phrase2)
            if p in self.pair2label and self.pair2label[p] != sample.y:
                logging.error('Adding pair with different label: phrase1="%s" phrase2="%s" old_label=%d old_source=%d new_label=%d new_source=%d',
                              sample.phrase1, sample.phrase2, self.pair2label[p], self.sample2source[p], sample.y, source)

            self.sample2source[p] = source
            self.pair2label[p] = sample.y

    def extend(self, samples, source):
        for sample in samples:
            self.append(sample, source)

    def get_pair_source(self, phrase1, phrase2):
        return self.sample2source[(phrase1, phrase2)]

    def __len__(self):
        return len(self.samples)

    def get_all_phrases(self):
        all_phrases = set()
        for sample in self.samples:
            all_phrases.add(sample.phrase1)
            all_phrases.add(sample.phrase2)
        return all_phrases

    def get_positive_samples(self):
        return filter(lambda s: s.y == 1, self.samples)

    def enum(self):
        return self.samples

    def count_1(self):
        return sum(sample.y == 1 for sample in self.samples)

    def count_0(self):
        return sum(sample.y == 0 for sample in self.samples)

    def permutation(self):
        self.samples = np.random.permutation(self.samples)


# настраиваем логирование в файл
ruchatbot.utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'prepare_synonymy_dataset.log'))

logging.info('Start "%s"', os.path.basename(__file__))

fcleaner = PhraseCleaner()

# Для генерации негативных сэмплов нам надо исключать
# вероятность попадания перефразировок в качестве негативных
# примеров для групп, содержащих более 2 вариантов. Поэтому
# каждую фразу пометим номером ее исходной группы.
phrase2group = dict()


# Грузим датасеты с перефразировками
samples = Samples()
igroup = 0
group = []
G = nx.Graph()  # для определения синонимичных фраз, записанных в разных группах.
all_positive_pairs = set()
all_negative_pairs = set()
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
                        samples.append(Sample(phrase1, phrase2, y), 1)
                        if phrase1 != phrase2:
                            # меняем местами сравниваемые фразы
                            samples.append(Sample(phrase2, phrase1, y), 2)

                        if y == 1 and phrase1 != phrase2:
                            G.add_edge(phrase1, phrase2)
                            all_positive_pairs.add((phrase1, phrase2))
                            all_positive_pairs.add((phrase2, phrase1))

                        if y == 0:
                            all_negative_pairs.add((phrase1, phrase2))
                            all_negative_pairs.add((phrase2, phrase1))

                group = []
        else:
            group.append(phrase)

logging.info('%d positive pairs, %d negative pairs have been loaded from "%s"',
             len(all_positive_pairs), len(all_negative_pairs), input_path)

# Для каждой фразы соберем список ее перефразировок, учитывая еще и описанные в разных группах.
phrase2synonyms = dict((phrase1, set(nx.algorithms.descendants(G, phrase1)))
                       for phrase1, _
                       in all_positive_pairs)

# Добавим перефразировки, которые на попали в одну группу, но определяются через связность в графе
n_added_positives = 0
for phrase1, synonyms in phrase2synonyms.items():
    for phrase2 in synonyms:
        p = (phrase1, phrase2)
        if p not in all_positive_pairs:
            samples.append(Sample(phrase1, phrase2, 1), 3)
            all_positive_pairs.add(p)
            n_added_positives += 1

        p = (phrase2, phrase1)
        if p not in all_positive_pairs:
            samples.append(Sample(phrase2, phrase1, 1), 4)
            all_positive_pairs.add(p)
            n_added_positives += 1

logging.info('%d positive pairs added from graph connectivity', n_added_positives)

# Из датасета для "антонимов" берем обязательные негативные примеры.
group = []
nb_antonyms = 0
nb_graph_antonyms = 0
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
                        p = (phrase1, phrase2)
                        if p not in all_negative_pairs:
                            samples.append(Sample(phrase1, phrase2, 0), 5)
                            all_negative_pairs.add(p)
                            nb_antonyms += 1

                        if phrase1 != phrase2:
                            # меняем местами сравниваемые фразы
                            p = (phrase2, phrase2)
                            if p not in all_negative_pairs:
                                samples.append(Sample(phrase2, phrase1, 0), 6)
                                all_negative_pairs.add(p)
                                nb_antonyms += 1

                        # Добавляем такой же антоним для всех синонимов первой фразы
                        if phrase1 in phrase2synonyms:
                            for synonym1 in phrase2synonyms[phrase1]:
                                if synonym1 != phrase1:
                                    p = (synonym1, phrase2)
                                    if p not in all_negative_pairs:
                                        samples.append(Sample(synonym1, phrase2, 0), 7)
                                        all_negative_pairs.add(p)
                                        nb_antonyms += 1

                group = []
        else:
            group.append(phrase)

logging.info('%d antonyms loaded from "contradictions.txt" and generated using graph', nb_antonyms)


# Из датасета "нерелевантные пары" берем пары фраз, которые гарантированно не являются синонимами
nb_nonrelevants = 0
with io.open(os.path.join(data_folder, 'nonrelevant_premise_questions.txt'), 'r', encoding='utf-8') as rdr:
    for line in rdr:
        parts = line.strip().split(u'|')
        if len(parts) == 2:
            phrase1 = fcleaner.process(parts[0])
            phrase2 = fcleaner.process(parts[1])
            p = (phrase1, phrase2)
            if p not in all_negative_pairs:
                samples.append(Sample(phrase1, phrase2, 0), 8)
                phrase2group[phrase1] = -1
                phrase2group[phrase2] = -1
                all_negative_pairs.add(p)
                nb_nonrelevants += 1
logging.info('%d nonrelevant pairs loaded from "nonrelevant_premise_questions.txt"', nb_nonrelevants)


# добавим некоторое кол-во идентичных пар.
all_phrases = samples.get_all_phrases()

duplicates = []
for phrase in all_phrases:
    duplicates.append(Sample(phrase, phrase, 1))

duplicates = np.random.permutation(duplicates)[:len(samples)//10]
for sample in duplicates:
    p = (sample.phrase1, sample.phrase2)
    if p not in all_positive_pairs:
        all_positive_pairs.add(p)
        samples.append(sample, 9)


# перестановочные дубликаты
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
                            p = group[i1], group[i2]
                            if p not in all_positive_pairs:
                                duplicates.append(Sample(group[i1], group[i2], 1))
                    group = []
            else:
                group.append(fcleaner.process(phrase))

# оставим кол-во дубликатов, сопоставимое с другими перефразировками
duplicates = np.random.permutation(duplicates)[:len(all_positive_pairs) // 4]
logging.info('%d duplicates with permutations loaded from "SENT*.duplicates.txt"', len(duplicates))

samples.extend(duplicates, 10)
all_positive_pairs.update((s.phrase1, s.phrase2) for s in duplicates)

all_phrases = list(samples.get_all_phrases())

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
    for phrase1 in all_phrases:
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
            p = (phrase1, phrase2)
            if p not in all_positive_pairs and p not in all_negative_pairs:
                neg_samples.add(phrase1, phrase2)

# Добавляем рандомные негативные сэмплы
for sample in samples.enum():
    igroup1 = phrase2group[sample.phrase1]
    n_neg = 0
    while n_neg < 10:
        neg_phrase = random.choice(all_phrases)
        if neg_phrase not in phrase2group:
            pass

        if phrase2group[neg_phrase] != igroup1:
            p = (sample.phrase1, neg_phrase)
            if p not in all_positive_pairs and p not in all_negative_pairs:
                neg_samples.add(sample.phrase1, neg_phrase)
                n_neg += 1

            p = (sample.phrase2, neg_phrase)
            if p not in all_positive_pairs and p not in all_negative_pairs:
                neg_samples.add(sample.phrase2, neg_phrase)
                n_neg += 1

# Сколько позитивных сэмплов
nb_1 = samples.count_1()

# ограничим кол-во негативных сэмплов
neg_samples = neg_samples.get_samples()
neg_samples = np.random.permutation(neg_samples)[:nb_1 * nb_neg_per_posit]
logging.info('%d negative samples added', len(neg_samples))
samples.extend(neg_samples, 11)
all_negative_pairs.update((s.phrase1, s.phrase2) for s in neg_samples)
samples.permutation()

nb_0 = samples.count_0()
logging.info('Final balance: nb_0=%d nb_1=%d', nb_0, nb_1)

max_wordseq_len = 0
for sample in samples.enum():
    for phrase in [sample.phrase1, sample.phrase2]:
        words = phrase.split(u' ')
        max_wordseq_len = max(max_wordseq_len, len(words))

logging.info('max_wordseq_len=%d', max_wordseq_len)

# сохраним получившийся датасет в CSV
logging.info(u'Storing result dataset with %d rows to "%s"', len(samples), output_filepath)
with codecs.open(output_filepath, 'w', 'utf-8') as wrt:
    # Заголовки делаем как у датасета relevancy моделей, чтобы их можно было использовать без переделок.
    wrt.write(u'premise\tquestion\trelevance\tweight\n')
    for sample in samples.enum():
        wrt.write(u'{}\t{}\t{}\t1\n'.format(sample.phrase1, sample.phrase2, sample.y))

# 16-03-2020 анализ частот символьных 3-грамм
shingle_len = 3
min_shingle_freq = 2
shingle2freq = collections.Counter()
for phrase in all_phrases:
    shingle2freq.update(ngrams(phrase, shingle_len))

# шинглы с частотой ниже пороговой
rare_shingles = set(shingle for shingle, freq in shingle2freq.items() if freq < min_shingle_freq)

# фразы с редкими шинглами
phrases_with_rare_shingles = set()
for phrase in all_phrases:
    if any((shingle in rare_shingles) for shingle in ngrams(phrase, shingle_len)):
        phrases_with_rare_shingles.add(phrase)

# теперь выводим отчет по редким шинглам и содержащим их фразам
with io.open(os.path.join(tmp_folder, 'synonymy_dataset_rare_shingles.txt'), 'w', encoding='utf-8') as wrt:
    for shingle, freq in sorted(shingle2freq.items(), key=lambda z: z[0]):
        if freq < min_shingle_freq:
            wrt.write('\n\nshingle={} freq={}\n'.format(shingle, freq))
            # ищем сэмплы с этим шинглом
            for phrase in phrases_with_rare_shingles:
                if shingle in ngrams(phrase, shingle_len):
                    wrt.write('{}\n'.format(phrase))

# Поищем слова, встречающиеся 1 раз
word2freq = collections.Counter()
for phrase in all_phrases:
    word2freq.update(phrase.split(' '))

rare_words = set(word for (word, freq) in word2freq.items() if freq == 1)
with io.open(os.path.join(tmp_folder, 'synonymy_dataset_rare_words.txt'), 'w', encoding='utf-8') as wrt:
    for phrase in all_phrases:
        words = phrase.split()
        for w in words:
            if w in rare_words:
                wrt.write('{:<20s} ===> {}\n\n'.format(w, phrase))

# Поищем несловарные и нечисловые токены
vocabulary = set()
with io.open(os.path.join(data_folder, 'dict/word2lemma.dat'), 'r', encoding='utf-8') as rdr:
    for line in rdr:
        fields = line.strip().split('\t')
        if len(fields) == 4:
            word = fields[0].lower().replace(' - ', '-')
            vocabulary.add(word)

oov_tokens = set()
for word, freq in word2freq.items():
    if word not in vocabulary and not is_int(word) and word not in '. ? ! : - , — – ) ( " \' « » „ “ ; …'.split():
        oov_tokens.add(word)

with io.open(os.path.join(tmp_folder, 'synonymy_dataset_oov_words.txt'), 'w', encoding='utf-8') as wrt:
    for phrase in all_phrases:
        words = phrase.split()
        for w in words:
            if w in oov_tokens:
                wrt.write('{:<20s} ===> {}\n\n'.format(w, phrase))




# --------------------------------------------------------------------------
# Теперь готовим датасет для модели детектора перефразировок с triplet loss
# Тут нам надо готовить триплеты (anchor, positive, negative)
logging.info('Start building dataset for triplet loss model of synonymy')

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


samples2 = [Sample2(s.phrase1, s.phrase2) for s in samples.get_positive_samples()]

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
for sample in samples2:
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
logging.info(u'Storing %d triplets to dataset "%s"', len(samples), output_filepath3)
with codecs.open(output_filepath3, 'w', 'utf-8') as wrt:
    # Заголовки делаем как у датасета relevancy моделей, чтобы их можно было использовать без переделок.
    wrt.write(u'anchor\tpositive\tnegative\n')
    for sample in samples3:
        wrt.write(u'{}\t{}\t{}\n'.format(sample.anchor, sample.positive, sample.negative))

logging.info('Finish "%s"', os.path.basename(__file__))


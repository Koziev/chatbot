# -*- coding: utf-8 -*-
"""
Готовим датасет для обучения модели, определяющей релевантность
вопроса и предпосылки, для проекта чатбота (https://github.com/Koziev/chatbot)

Используется несколько входных датасетов.
qa.txt - основной датасет, содержит релевантные вопросы и предпосылки
paraphrases.txt - отсюда получаем релевантные и нерелевантные перефразировки
questions.txt - список вопросов для негативного сэмплинга
и т.д.

Создаваемый датасет в формате CSV представляет из себя четыре колонки.
Первые две - текст сравниваемых фраз.
Третья содержит целое число:
0 - не релевантны
1 - есть полная семантическая релевантность
Четвертая - вес сэмпла, 1 для автоматически сгенерированных, >1 для сэмплов
из вручную сформированных файлов.

24-12-2019 Добавлена аугментация датасета с помощью замен на синонимичные фразы, взятые
           из paraphrases.txt

04-07-2020 Доработка загрузки негативных сэмплов из файла nonrelevant_premise_questions.txt
"""

from __future__ import division  # for python2 compatability
from __future__ import print_function

import codecs
import collections
import itertools
import os
import random
import tqdm
import numpy as np
import io
import pandas as pd

import networkx as nx

from rutokenizer import Tokenizer
from preparation.corpus_searcher import CorpusSearcher


USE_AUTOGEN = True  # добавлять ли сэмплы из автоматически сгенерированных датасетов

HANDCRAFTED_WEIGHT = 1  # вес для сэмплов, которые в явном виде созданы вручную
AUTOGEN_WEIGHT = 1  # вес для синтетических сэмплов, сгенерированных автоматически

# Автоматически сгенерированных сэмплов очень много, намного больше чем вручную
# составленных, поэтому ограничим количество паттернов для каждого типа автоматически
# сгенерированных.
MAX_NB_AUTOGEN = 1000  # макс. число автоматически сгенерированных сэмплов одного типа

ADD_SIMILAR_NEGATIVES = False  # негативные вопросы подбирать по похожести к предпосылке (либо чисто рандомные)

n_negative_per_positive = 10

tmp_folder = '../../tmp'
data_folder = '../../data'
paraphrases_paths = ['../../data/paraphrases.txt']
qa_paths = [('qa.txt', HANDCRAFTED_WEIGHT, 10000000)]

if USE_AUTOGEN:
    qa_paths.extend([('current_time_pqa.txt', AUTOGEN_WEIGHT, 1000000),
                     ('premise_question_answer6.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer5.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer4.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer4_1s.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer4_2s.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer5_1s.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN),
                     ('premise_question_answer5_2s.txt', AUTOGEN_WEIGHT, MAX_NB_AUTOGEN)
                     ])
questions_path = '../../data/questions.txt'

include_repeats = True

stop_words = set(u'не ни ль и или ли что какой же ж какая какие сам сама сами само был были было есть'.split())
stop_words.update(u'о а в на у к с со по ко мне нам я он она над за из от до'.split())

# ---------------------------------------------------------------


class PhraseCleaner:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.load()

    def process(self, phrase):
        return u' '.join(self.tokenizer.tokenize(phrase.lower()))



def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(shingles1, shingles2):
    return float(len(shingles1 & shingles2)) / float(len(shingles1 | shingles2))


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
    line = line.replace('  ', ' ')
    line = line.strip().lower()
    line = ru_sanitize(line)
    return line


class ResultantDataset(object):
    def __init__(self):
        self.str_pairs = []  # предпосылки и вопросы
        self.relevancy = []  # релевантность вопросов и предпосылок в парах
        self.weights = []  # вес сэмпла
        self.added_pairs_set = set()  # для предотвращения повторов

    def add_pair(self, x1, x2, rel, weight):
        s1 = x1.replace(u'\t', u'').strip()
        s2 = x2.replace(u'\t', u'').strip()
        s12 = s1 + '|' + s2
        if s12 not in self.added_pairs_set:
            self.added_pairs_set.add(s12)
            self.str_pairs.append((s1, s2))
            self.relevancy.append(rel)
            self.weights.append(weight)
            return True

        return False

    def positive_count(self):
        return sum(self.relevancy)

    def save_csv(self, filepath):
        # сохраним получившийся датасет в CSV
        with codecs.open(filepath, 'w', 'utf-8') as wrt:
            wrt.write(u'premise\tquestion\trelevance\tweight\n')
            for (s1, s2), r, w in np.random.permutation(list(zip(self.str_pairs, self.relevancy, self.weights))):
                wrt.write(u'{}\t{}\t{}\t{}\n'.format(s1, s2, r, w))

    def print_stat(self):
        print('Total number of samples={}'.format(len(self.str_pairs)))

        for y in range(max(self.relevancy) + 1):
            print('rel={} number of samples={}'.format(y, sum(1 for z in self.relevancy if z == y)))

        weight2count = collections.Counter()
        for w in self.weights:
            weight2count[w] += 1

        print('premise  max len={}'.format(max(map(lambda z: len(z[0]), self.str_pairs))))
        print('question max len={}'.format(max(map(lambda z: len(z[1]), self.str_pairs))))

    def list_positives(self):
        for pair, rel in zip(self.str_pairs, self.relevancy):
            if rel == 1.0:
                yield pair

    def remove_redundant_negatives(self, nb_negative):
        pos_sample_indexes = [i for i in range(len(self.relevancy)) if self.relevancy[i] == 1]
        neg_sample_indexes = [i for i in range(len(self.relevancy)) if self.relevancy[i] != 1]
        neg_sample_indexes = np.random.permutation(neg_sample_indexes)
        neg_sample_indexes = neg_sample_indexes[:nb_negative]

        # берем все позитивные примеры
        str_pairs1 = [self.str_pairs[i] for i in pos_sample_indexes]
        relevancy1 = [self.relevancy[i] for i in pos_sample_indexes]
        weights1 = [self.weights[i] for i in pos_sample_indexes]

        # оставшиеся после усечения негативные примеры
        str_pairs0 = [self.str_pairs[i] for i in neg_sample_indexes]
        relevancy0 = [self.relevancy[i] for i in neg_sample_indexes]
        weights0 = [self.weights[i] for i in neg_sample_indexes]

        self.str_pairs = list(itertools.chain(str_pairs1, str_pairs0))
        self.relevancy = list(itertools.chain(relevancy1, relevancy0))
        self.weights = list(itertools.chain(weights1, weights0))


if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokenizer.load()
    random_questions = CorpusSearcher()
    random_facts = CorpusSearcher()

    # прочитаем список случайных вопросов из заранее сформированного файла
    # (см. код на C# https://github.com/Koziev/chatbot/tree/master/CSharpCode/ExtractFactsFromParsing
    # и результаты его работы https://github.com/Koziev/NLP_Datasets/blob/master/Samples/questions4.txt)
    print('Loading random questions and facts...')
    with codecs.open(questions_path, 'r', 'utf-8') as rdr:
        for line in rdr:
            if len(line) < 40:
                question = line.strip()
                question = ru_sanitize(u' '.join(tokenizer.tokenize(question.lower())))
                random_questions.add_phrase(normalize_qline(question))

    # Прочитаем список случайных фактов, чтобы потом генерировать отрицательные паттерны
    for facts_path in ['paraphrases.txt', 'facts4.txt', 'facts5.txt', 'facts6.txt', ]:
        with codecs.open(os.path.join(data_folder, facts_path), 'r', 'utf-8') as rdr:
            n = 0
            for line in rdr:
                s = line.strip()
                if s:
                    if s[-1] == u'?':
                        random_questions.add_phrase(normalize_qline(s))
                    else:
                        random_facts.add_phrase(normalize_qline(s))

                    n += 1
                    if n > 2000000:
                        break

    print('{} random facts in set'.format(len(random_facts)))
    print('{} random questions in set'.format(len(random_questions)))
    # ------------------------------------------------------------------------

    res_dataset = ResultantDataset()

    lines = []
    posit_pairs_count = 0
    negat_pairs_count = 0
    random_negat_pairs_count = 0

    # Из отдельного файла загрузим список нерелевантных пар предпосылка-вопрос.
    manual_negatives_pq = collections.defaultdict(list)
    manual_negatives_qp = collections.defaultdict(list)
    with codecs.open(os.path.join(data_folder, 'nonrelevant_premise_questions.txt'), 'r', 'utf-8') as rdr:
        for line in rdr:
            line = line.strip()
            if line:
                tx = line.split('|')
                if len(tx) == 2:
                    premise = normalize_qline(tx[0])
                    question = normalize_qline(tx[1])

                    manual_negatives_pq[premise].append(question)
                    manual_negatives_qp[question].append(premise)
                elif len(tx) == 1:
                    # Второй формат, аналогичный негативным синонимам

                    # Может быть задан один вопрос и к нему много нерелевантных предпосылок,
                    # или одна предпосылка и к ней много нерелевантных вопросов.
                    # Определяем по последнему символу первой строки
                    if tx[0].endswith('?'):
                        # вопрос и к нему много нерелевантных предпосылок
                        question = normalize_qline(tx[0])
                        for line in rdr:
                            if line.startswith('(-)'):
                                premise = normalize_qline(line.replace('(-)', '').strip())
                                manual_negatives_qp[question].append(premise)
                                manual_negatives_pq[premise].append(question)
                            else:
                                break
                    else:
                        # предпосылка и к ней много нерелевантных вопросов
                        premise = normalize_qline(tx[0])
                        for line in rdr:
                            if line.startswith('(-)'):
                                question = normalize_qline(line.replace('(-)', '').strip())
                                manual_negatives_qp[question].append(premise)
                                manual_negatives_pq[premise].append(question)
                            else:
                                break


    for premise, questions in manual_negatives_pq.items():
        for question in questions:
            res_dataset.add_pair(premise, question, 0, 1)


    # Загрузим датасет с перефразировками фраз, чтобы выполнить аугментацию pq-датасета синонимами.
    G = nx.Graph()  # для определения синонимичных фраз, записанных в разных группах.
    group = []
    all_positive_pairs = set()
    igroup = 0
    fcleaner = PhraseCleaner()
    phrase2group = dict()

    for p in paraphrases_paths:
        with io.open(p, 'r', encoding='utf-8') as rdr:
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
                                if y == 1 and phrase1 != phrase2:
                                    G.add_edge(phrase1, phrase2)
                                    all_positive_pairs.add((phrase1, phrase2))
                                    all_positive_pairs.add((phrase2, phrase1))

                        group = []
                else:
                    group.append(phrase)

    # Для каждой фразы соберем список ее перефразировок, учитывая еще и описанные в разных группах.
    phrase2synonyms = dict((phrase1, set(nx.algorithms.descendants(G, phrase1)))
                           for phrase1, _
                           in all_positive_pairs)


    # Теперь релевантные пары предпосылка-вопрос.
    for qa_path, qa_weight, max_samples in qa_paths:
        print('Parsing {}'.format(qa_path), end='\r')
        premise_questions = []
        posit_pairs_count = 0
        negat_pairs_count = 0
        with codecs.open(os.path.join(data_folder, qa_path), "r", "utf-8") as inf:
            loading_state = 'T'

            text = []
            questions = []

            for line in inf:
                line = line.strip()

                if line.startswith(u'T:'):
                    if loading_state == 'T':
                        text.append(normalize_qline(line))
                    else:
                        # закончился парсинг предыдущего блока из текста (предпосылки),
                        # вопросов и ответов.

                        # Из загруженных записей добавим пары в обучающий датасет
                        if len(text) == 1:
                            premise_questions.append((text[0], questions))

                            for premise in text:
                                for question in questions:
                                    if res_dataset.add_pair(premise, question, 1, qa_weight):
                                        posit_pairs_count += 1

                                    premise_syns = set([premise]) | phrase2synonyms.get(fcleaner.process(premise), set())
                                    premise_syns = [s for s in premise_syns if u'?' not in s and u'ли' not in s]
                                    question_syns = set([question]) | phrase2synonyms.get(fcleaner.process(question), set())
                                    for premise1 in premise_syns:
                                        for question1 in question_syns:
                                            if res_dataset.add_pair(premise1, question1, 1, qa_weight):
                                                posit_pairs_count += 1

                        loading_state = 'T'
                        questions = []
                        text = [normalize_qline(line)]

                        if posit_pairs_count >= max_samples:
                            break

                elif line.startswith(u'Q:'):
                    loading_state = 'Q'
                    questions.append(normalize_qline(line))

        print('{:<40s} ==> posit_pairs_count={}'.format(qa_path, posit_pairs_count))

    # Добавляем негативные сэмплы
    for premise, question in tqdm.tqdm(res_dataset.list_positives(), total=res_dataset.positive_count(), desc='Adding negative'):
        # Добавим несколько нерелевантных вопросов и предпосылок, используя части релевантного сэмпла
        neg_2_add = n_negative_per_positive

        if ADD_SIMILAR_NEGATIVES:
            # Берем ближайшие случайные вопросы и предпосылки
            for neg_question in random_questions.find_similar(premise, neg_2_add // 2 if neg_2_add > 0 else 1):
                res_dataset.add_pair(premise, neg_question, 0, AUTOGEN_WEIGHT)
                negat_pairs_count += 1
                neg_2_add -= 1

            for neg_premise in random_facts.find_similar(question, neg_2_add):
                res_dataset.add_pair(neg_premise, question, 0, AUTOGEN_WEIGHT)
                negat_pairs_count += 1
                neg_2_add -= 1

        # Добавим случайные нерелевантные вопросы и предпосылки
        if neg_2_add > 0:
            for neg_question in random_questions.get_random(neg_2_add // 2 if neg_2_add > 0 else 1):
                res_dataset.add_pair(premise, neg_question, 0, AUTOGEN_WEIGHT)
                negat_pairs_count += 1
                neg_2_add -= 1

            if neg_2_add > 0:
                for neg_premise in random_facts.get_random(neg_2_add):
                    res_dataset.add_pair(neg_premise, question, 0, AUTOGEN_WEIGHT)
                    negat_pairs_count += 1
                    neg_2_add -= 1

        assert (neg_2_add == 0)

    # ---------------------------------------------------------------------------

    # Добавляем перестановочные перефразировки.
    # Подготовка датасетов с перестановочными перефразировками выполняется
    # C# кодом, находящимся здесь: https://github.com/Koziev/NLP_Datasets/tree/master/ParaphraseDetection
    if False:
        srcpaths = ['SENT4.duplicates.txt', 'SENT5.duplicates.txt', 'SENT6.duplicates.txt']

        # кол-во перестановочных перефразировок одной длины,
        # чтобы в итоге кол-во перестановочных пар не превысило число
        # явно заданных положительных пар.
        nb_permut = res_dataset.positive_count() / len(srcpaths)

        print('nb_permut={}'.format(nb_permut))
        total_permutations = 0
        include_repeats = False  # включать ли нулевые перефразировки - когда левая и правая части идентичны
        emitted_perm = set()

        for srcpath in srcpaths:
            print('source=', srcpath)
            lines = []
            with codecs.open(os.path.join(data_folder, srcpath), "r", "utf-8") as inf:
                nperm = 0
                for line in inf:
                    line = line.strip()
                    if len(line) == 0:
                        if len(lines) > 1:

                            for i1 in range(len(lines)):
                                for i2 in range(len(lines)):
                                    if i1 == i2 and not include_repeats:
                                        continue
                                    k1 = lines[i1].strip() + u'|' + lines[i2].strip()
                                    k2 = lines[i2].strip() + u'|' + lines[i1].strip()
                                    if k1 not in emitted_perm and k2 not in emitted_perm:
                                        emitted_perm.add(k1)
                                        emitted_perm.add(k2)

                                        res_dataset.add_pair(lines[i1], lines[i2], 1, AUTOGEN_WEIGHT)
                                        total_permutations += 1
                                        nperm += 1
                                        if nperm > nb_permut:
                                            break

                        lines = []
                    else:
                        lines.append(normalize_qline(line))

        print('total_permutations={}'.format(total_permutations))

    # ---------------------------------------------------------------------------

    # Теперь сокращаем кол-во негативных сэмплов
    # nb_negative = res_dataset.positive_count() * n_negative_per_positive
    # res_dataset.remove_redundant_negatives(nb_negative)

    # Выведем итоговую статистику
    res_dataset.print_stat()

    # сохраним получившийся датасет в CSV
    print('Storing dataset..')
    res_dataset.save_csv(os.path.join(data_folder, 'premise_question_relevancy.csv'))

    # ------------------------------------------------------------
    # Генерация датасета для модели частичной релевантности (сэмплы с 2-мя предпосылками)
    p2qa_samples = []
    with io.open(os.path.join(data_folder, 'qa_multy.txt'), 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            line = normalize_qline(line)
            if line:
                lines.append(line)
            else:
                if len(lines) == 4:
                    premises = lines[:-2]
                    question = lines[-2]
                    answer = lines[-1]
                    p2qa_samples.append((premises, question, answer))

                lines = []

    print('{} p2qa samples loaded from qa_multi.txt'.format(len(p2qa_samples)))
    partial_p2q = dict()
    random_premises = list(set(s[0] for s in res_dataset.str_pairs))

    for premises, question, answer in p2qa_samples:
        for premise in premises:
            partial_p2q[(premise, question)] = 1

            # добавим негативный пример, рандомно выбирая предпосылку
            neg_premise = random.choice(random_premises)
            k = (neg_premise, question)
            if k not in partial_p2q:
                partial_p2q[k] = 0

    filepath = os.path.join(data_folder, 'partial_premise_question_relevancy.tsv')
    print('Writing {} samples to "{}"'.format(len(partial_p2q), filepath))
    with io.open(filepath, 'w', encoding='utf-8') as wrt:
        wrt.write(u'premise\tquestion\trelevance\tweight\n')
        for (premise, question), r in partial_p2q.items():
            wrt.write(u'{}\t{}\t{}\t{}\n'.format(premise, question, r, 1))

    # Датасет для тренировки модели релевантности 2P<==>Q
    added_p2q = set()
    filepath = os.path.join(data_folder, '2premises_question_relevancy.tsv')
    with io.open(filepath, 'w', encoding='utf-8') as wrt:
        wrt.write(u'premise1\tpremise2\tquestion\trelevance\n')
        for premises, question, answer in p2qa_samples:
            k = premises[0], premises[1], question
            if k not in added_p2q:
                wrt.write('{}\t{}\t{}\t1\n'.format(premises[0], premises[1], question))
                added_p2q.add(k)

            k = premises[1], premises[0], question
            if k not in added_p2q:
                wrt.write('{}\t{}\t{}\t1\n'.format(premises[1], premises[0], question))
                added_p2q.add(k)

            # Заменяем рандомно одну или обе предпосылки - получаем нерелевантный сэмпл.
            premise1 = random.choice(random_premises)
            wrt.write('{}\t{}\t{}\t0\n'.format(premise1, premises[1], question))

            premise2 = random.choice(random_premises)
            wrt.write('{}\t{}\t{}\t0\n'.format(premises[0], premise2, question))

            premise1 = random.choice(random_premises)
            premise2 = random.choice(random_premises)
            wrt.write('{}\t{}\t{}\t0\n'.format(premise1, premise2, question))

        # добавка ручных негативных сэмплов
        with io.open(os.path.join(data_folder, 'nonrelevant_2premises_questions.txt'), 'r', encoding='utf-8') as rdr:
            lines = []
            for line in rdr:
                line = line.strip()
                if line:
                    lines.append(line)
                else:
                    if len(lines) == 3:
                        premise1 = lines[0]
                        premise2 = lines[1]
                        question = lines[2]
                        wrt.write('{}\t{}\t{}\t0\n'.format(premise1, premise2, question))

                    lines = []

    df = pd.read_csv(filepath, delimiter='\t')
    n0 = df[df['relevance'] == 0].shape[0]
    n1 = df[df['relevance'] == 1].shape[0]
    print('{} samples stored in "{}": n0={}, n1={}'.format(n0+n1, filepath, n0, n1))

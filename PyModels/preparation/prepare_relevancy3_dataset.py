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
import os
import tqdm

try:
    from itertools import izip as zip
except ImportError:
    pass

from utils.tokenizer import Tokenizer
from preparation.corpus_searcher import CorpusSearcher


# Путь к создаваемому датасету для модели детектора на базе triplet loss
output_filepath3 = '../../data/relevancy_dataset3.csv'


USE_AUTOGEN = True  # добавлять ли сэмплы из автоматически сгенерированных датасетов

HANDCRAFTED_WEIGHT = 1  # вес для сэмплов, которые в явном виде созданы вручную
AUTOGEN_WEIGHT = 1  # вес для синтетических сэмплов, сгенерированных автоматически

FILTER_KNOWN_WORDS = True  # не брать негативный сэмпл, если в нем есть неизвестные слова

# Автоматически сгенерированных сэмплов очень много, намного больше чем вручную
# составленных, поэтому ограничим количество паттернов для каждого типа автоматически
# сгенерированных.
MAX_NB_AUTOGEN = 100000  # макс. число автоматически сгенерированных сэмплов одного типа

ADD_SIMILAR_NEGATIVES = False  # негативные вопросы подбирать по похожести к предпосылке (либо чисто рандомные)

n_negative_per_positive = 5

tmp_folder = '../../tmp'
data_folder = '../../data'
paraphrases_paths = ['../../data/paraphrases.txt', '../../data/contradictions.txt']
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
    return float(len(shingles1 & shingles2)) / float(len(shingles1 | shingles2))


# ---------------------------------------------------------------

tokenizer = Tokenizer()
tokenizer.load()

random_questions = CorpusSearcher()
random_facts = CorpusSearcher()

qwords = set(u'кто кому ком когда где зачем почему откуда куда как сколько что чем чему чего'.split())

# прочитаем список случайных вопросов из заранее сформированного файла
# (см. код на C# https://github.com/Koziev/chatbot/tree/master/CSharpCode/ExtractFactsFromParsing
# и результаты его работы https://github.com/Koziev/NLP_Datasets/blob/master/Samples/questions4.txt)
print('Loading random questions and facts')
with codecs.open(questions_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        if len(line) < 40:
            question = line.strip()
            words = tokenizer.tokenize(question)
            if any((w in qwords) for w in words):
                question = ru_sanitize(u' '.join(words))
                random_questions.add_phrase(question)

for facts_path in ['paraphrases.txt', 'facts4.txt', 'facts5.txt', 'facts6.txt']:
    with codecs.open(os.path.join(data_folder, facts_path), 'r', 'utf-8') as rdr:
        n = 0
        for line in rdr:
            s = line.strip()
            if s:
                if s[-1] == u'?':
                    words = tokenizer.tokenize(question)
                    if any((w in qwords) for w in words):
                        question = ru_sanitize(u' '.join(words))
                        random_questions.add_phrase(question)
            n += 1
            if n > 2000000:
                break


for q_path in qa_paths:
    with codecs.open(os.path.join(data_folder, q_path[0]), 'r', 'utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s.startswith('Q:'):
                question = normalize_qline(s)
                words = tokenizer.tokenize(question)
                if any((w in qwords) for w in words):
                    question = u' '.join(words)
                    random_questions.add_phrase(question)

# Прочитаем список случайных фактов
for facts_path in ['paraphrases.txt', 'facts4.txt', 'facts5.txt', 'facts6.txt', ]:
    with codecs.open(os.path.join(data_folder, facts_path), 'r', 'utf-8') as rdr:
        n = 0
        for line in rdr:
            s = line.strip()
            if s:
                if s[-1] != u'?':
                    random_facts.add_phrase(normalize_qline(s))

                n += 1
                if n > 2000000:
                    break

print('Random facts pool size={}'.format(len(random_facts)))
print('Random questions pool size={}'.format(len(random_questions)))


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
manual_negatives_pq = dict()
manual_negatives_qp = dict()
with codecs.open(os.path.join(data_folder, 'nonrelevant_premise_questions.txt'), 'r', 'utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        if line:
            tx = line.split('|')
            if len(tx) == 2:
                premise = normalize_qline(tx[0])
                question = normalize_qline(tx[1])
                if premise not in manual_negatives_pq:
                    manual_negatives_pq[premise] = [question]
                else:
                    manual_negatives_pq[premise].append(question)

                if question not in manual_negatives_qp:
                    manual_negatives_qp[question] = [premise]
                else:
                    manual_negatives_qp[question].append(premise)


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


# Второй проход по списку пар предпосылка-вопрос.
# Для каждой пары подбираем негативный вопрос.
print('Adding negative questions to samples...')
samples3 = []
n3 = 0
for premise, question in tqdm.tqdm(samples2, desc='Adding negatives', total=len(samples2)):
    # Для предпосылки есть заданные вручную негативные вопросы?
    if premise in manual_negatives_pq:
        for nonrelevant_question in manual_negatives_pq[premise]:
            pq = premise + u'|' + nonrelevant_question
            if pq not in all_pq:
                samples3.append(Sample3(premise, question, nonrelevant_question))
                all_pq.add(pq)

    if question in manual_negatives_qp:
        for nonrelevant_premise in manual_negatives_qp[question]:
            pq = nonrelevant_premise + u'|' + question
            if pq not in all_pq:
                samples3.append(Sample3(question, premise, nonrelevant_premise))
                all_pq.add(pq)


    neg_2_add = n_negative_per_positive

    if ADD_SIMILAR_NEGATIVES:
        # Берем ближайшие случайные вопросы
        for neg_question in random_questions.find_similar(premise, neg_2_add // 2 if neg_2_add > 0 else 1):
            pq = premise + u'|' + neg_question
            if pq not in all_pq:
                samples3.append(Sample3(premise, question, neg_question))
                all_pq.add(pq)
                neg_2_add -= 1

        for neg_premise in random_facts.find_similar(question, neg_2_add):
            pq = neg_premise + u'|' + question
            if pq not in all_pq:
                samples3.append(Sample3(question, premise, neg_premise))
                all_pq.add(pq)
                neg_2_add -= 1

    # Добавим случайные нерелевантные вопросы
    if neg_2_add > 0:
        for neg_question in random_questions.get_random(neg_2_add // 2 if neg_2_add > 0 else 1):
            pq = premise + u'|' + neg_question
            if pq not in all_pq:
                samples3.append(Sample3(premise, question, neg_question))
                all_pq.add(pq)
                neg_2_add -= 1

        for neg_premise in random_facts.get_random(neg_2_add):
            pq = neg_premise + u'|' + question
            if pq not in all_pq:
                samples3.append(Sample3(question, premise, neg_premise))
                all_pq.add(pq)
                neg_2_add -= 1

    #assert (neg_2_add == 0)


print(u'Storing {} triplets to dataset "{}"'.format(len(samples3), output_filepath3))
with codecs.open(output_filepath3, 'w', 'utf-8') as wrt:
    wrt.write(u'anchor\tpositive\tnegative\n')
    for sample in samples3:
        wrt.write(u'{}\t{}\t{}\n'.format(sample.anchor, sample.positive, sample.negative))

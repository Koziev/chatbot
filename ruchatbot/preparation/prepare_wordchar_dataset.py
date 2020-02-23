# -*- coding: utf-8 -*-
"""
Подготовка списка слов для тренировки модели встраиваний слова wordchar2vector и
и других моделей, где нужно оптимизировать данные под наши датасеты.

(c) by Koziev Ilya для чат-бота https://github.com/Koziev/chatbot
"""

from __future__ import print_function

import codecs
import io
import itertools
import os
from sys import platform
import pandas as pd
import csv
import yaml

from ruchatbot.utils.tokenizer import Tokenizer

result_path = '../../tmp/known_words.txt'  # путь к файлу, где будет сохранен полный список слов для обучения
result2_path = '../../tmp/dataset_words.txt'  # путь к файлу со списком слов, которые употребляются в датасетах чатбота

data_folder = '../../data'

n_misspelling_per_word = 0  # кол-во добавляемых вариантов с опечатками на одно исходное слово


# Из этого текстового файла возьмем слова, на которых будем тренировать модель встраивания.
if platform == "win32":
    corpus_path = r'f:\Corpus\word2vector\ru\SENTx.corpus.w2v.txt'
else:
    corpus_path = os.path.expanduser('~/corpora/Corpus/word2vector/ru/SENTx.corpus.w2v.txt')

paraphrases_path = '../../data/premise_question_relevancy.csv'
synonymy_path = '../../data/synonymy_dataset.csv'
synonymy3_path = '../../data/synonymy_dataset3.csv'
pqa_path = '../../data/premise_question_answer.csv'
pqa_multy_path = '../../data/qa_multy.txt'
eval_path = '../../data/evaluate_relevancy.txt'
syntax_validator_path = '../../data/syntax_validator_dataset.csv'
premises = ['../../data/profile_facts_1.dat', '../../data/profile_facts_2.dat']
interpretations = ['../../data/interpretation_auto_4.txt',
                   '../../data/interpretation_auto_5.txt',
                   '../../data/interpretation.txt',
                   '../../data/entity_extraction.txt',
                   '../../data/intents.txt']

postagger_corpora = ['/home/inkoziev/polygon/rupostagger/tmp/samples.dat']
yaml_path = '../../data/rules.yaml'


goodchars = set(u'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ' +
                u'1234567890' +
                u'+.,-?!()[]{}*<>$&=~№/\\«»%:;|#"\'°')

letters = set(u'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')


stop_words = {u'_num_'}


lexicon_words = set()


def is_punkt(c):
    return c in u'+.,-?!()[]{}*<>$&=~№/\\«»%:;|#" \'’–'


def normalize_word(word):
    return word.lower().replace(u'ё', u'е')


def collect_strings(d):
    res = []

    if isinstance(d, str):
        if u'[' not in d and u']' not in d:
            res.append(d)
    elif isinstance(d, list):
        for item in d:
            res.extend(collect_strings(item))
    elif isinstance(d, dict):
        for k, node in d.items():
            res.extend(collect_strings(node))

    return res


def is_good_word(word):
    if word in stop_words or word.startswith(u' ') or word == u'' or len(word) > 28 or len(word) == 0:
        return False

    if len(word) > 1:
        if is_punkt(word[0]):
            # В датасетах попадаются мусорные токены типа ((((, которые нет
            # смысла сохранять для тренировки wc2v модели.
            return False

    if any(c not in goodchars for c in word):
        return False

    return True


tokenizer = Tokenizer()
tokenizer.load()

known_words = set()
dataset_words = set()

with io.open(os.path.join(data_folder, 'dict/word2lemma.dat'), 'r', encoding='utf-8') as rdr:
    for line in rdr:
        tx = line.replace(u'\ufeff', '').strip().split('\t')
        if len(tx) > 1:
            word = tx[0].lower().replace(' - ', '-')
            if word[0] in letters:
                lexicon_words.add(word)

for corpus in postagger_corpora:
    print(u'Processing {}'.format(corpus))
    with codecs.open(corpus, 'r', 'utf-8') as rdr:
        for line in rdr:
            line = line.strip()
            if line:
                tx = line.split('\t')
                word = normalize_word(tx[1])
                known_words.add(word)
                dataset_words.add(word)

print('Parsing {}'.format(yaml_path))
with io.open(yaml_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
    strings = collect_strings(data)
    for phrase in strings:
        phrase = phrase.strip()
        if u'_' not in phrase and any((c in u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя') for c in phrase):
            words = tokenizer.tokenize(phrase)
            known_words.update(words)
            dataset_words.update(words)

# Берем слова из большого текстового файла, на котором тренируется w2v модели.
print('Parsing {}'.format(corpus_path))
with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
    line_count = 0
    for line0 in rdr:
        line = line0.strip()
        words = [normalize_word(w) for w in line.split(u' ')]
        known_words.update(words)
        line_count += 1
        if line_count > 5000000:
            break

# Добавим слова из основного тренировочного датасета
print('Parsing {}'.format(paraphrases_path))
df = pd.read_csv(paraphrases_path, encoding='utf-8', delimiter='\t', quoting=3)
for phrase in itertools.chain(df['premise'].values, df['question'].values):
    words = tokenizer.tokenize(phrase.lower())
    known_words.update(words)
    dataset_words.update(words)

print('Parsing {}'.format(synonymy_path))
df = pd.read_csv(synonymy_path, encoding='utf-8', delimiter='\t', quoting=3)
for phrase in itertools.chain(df['premise'].values, df['question'].values):
    words = tokenizer.tokenize(phrase.lower())
    known_words.update(words)
    dataset_words.update(words)

print('Parsing {}'.format(synonymy3_path))
df = pd.read_csv(synonymy3_path, encoding='utf-8', delimiter='\t', quoting=3)
for phrase in itertools.chain(df['anchor'].values, df['positive'].values, df['negative'].values):
    words = tokenizer.tokenize(phrase.lower())
    known_words.update(words)
    dataset_words.update(words)

print('Parsing {}'.format(pqa_path))
df = pd.read_csv(pqa_path, encoding='utf-8', delimiter='\t', quoting=3)
for phrase in itertools.chain(df['premise'].values, df['question'].values, df['answer'].values):
    words = tokenizer.tokenize(phrase)
    known_words.update(words)
    dataset_words.update(words)

print('Parsing {}'.format(syntax_validator_path))
df = pd.read_csv(syntax_validator_path, encoding='utf-8', delimiter='\t', quoting=csv.QUOTE_NONE)
for phrase in df['sample'].values:
    words = phrase.split()
    known_words.update(words)
    dataset_words.update(words)

df = pd.read_csv('../../data/entities_dataset.tsv', encoding='utf-8', delimiter='\t', quoting=3)
for phrase in df['phrase'].values:
    words = tokenizer.tokenize(phrase.lower())
    known_words.update(words)
    dataset_words.update(words)

df = pd.read_csv('../../data/relevancy_dataset3.csv', encoding='utf-8', delimiter='\t', quoting=3)
for phrase in itertools.chain(df['anchor'].values, df['positive'].values, df['negative'].values):
    words = tokenizer.tokenize(phrase.lower())
    known_words.update(words)
    dataset_words.update(words)

with codecs.open('../../data/answer_relevancy_dataset.dat', 'r', 'utf-8') as rdr:
    for line in rdr:
        words = line.strip().split()
        known_words.update(words)
        dataset_words.update(words)

# Добавим слова, которые употребляются в датасете для оценки
print('Parsing {}'.format(eval_path))
with codecs.open(eval_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        phrase = line.replace(u'T:', u'').replace(u'Q:', u'').strip()
        words = tokenizer.tokenize(phrase)
        known_words.update(words)
        dataset_words.update(words)

# Добавим слова, которые употребляются в датасете с выводами
print('Parsing {}'.format(pqa_multy_path))
with codecs.open(pqa_multy_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        phrase = line.replace(u'T:', u'').replace(u'Q:', u'').replace(u'A:', u'').strip()
        words = tokenizer.tokenize(phrase)
        known_words.update(words)
        dataset_words.update(words)

for p in premises:
    print('Parsing {}'.format(p))
    with codecs.open(p, 'r', 'utf-8') as rdr:
        for line in rdr:
            phrase = line.strip()
            if phrase.startswith('#'):
                continue
            words = tokenizer.tokenize(phrase)
            known_words.update(words)
            dataset_words.update(words)

# Датасеты интерпретации
# Датасеты определения интента и выделения сущностей
for p in interpretations:
    print('Parsing {}'.format(p))
    with codecs.open(p, 'r', 'utf-8') as rdr:
        for line in rdr:
            phrase2 = line.strip()
            if phrase2.startswith('#') or phrase2.startswith('entity'):
                continue
            for phrase in phrase2.split('|'):
                words = tokenizer.tokenize(phrase)
                known_words.update(words)
                dataset_words.update(words)


print('There are {} known words, {} dataset words'.format(len(known_words), len(dataset_words)))

with codecs.open(result_path, 'w', 'utf-8') as wrt:
    for word in sorted(known_words):
        if is_good_word(word):
            wrt.write(u'{}\n'.format(word))

with codecs.open(result2_path, 'w', 'utf-8') as wrt:
    for word in sorted(dataset_words):
        if is_good_word(word):
            wrt.write(u'{}\n'.format(word))

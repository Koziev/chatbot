# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки модели определения intent'а в NLU движке чатбота
27.06.2019 - формат датасета приведен в соответствие с RASA

18.06.2020 из датасета интентов в отдельные классификации вынесены оскорбления, сентимент и направленность,
           поэтому для них формируются отдельные датасеты и тренируются свои классификаторы.

31.07.2020 Из нескольких файлов собирается набор фраз для пакетного прогона через натренированные модели, и
           сохраняется в  tmp/intents_batch_query.txt

02.11.2020 Добавлен детектор абракадабры. Сделан генератор искусственной абракадабры через рандочные цепочки токенов,
           получемых при обучении youtokentome модели

29.11.2020 В датасет оскорбительных фраз добавлены примеры из https://www.kaggle.com/alexandersemiletov/toxic-russian-comments
"""

from __future__ import division  # for python2 compatability
from __future__ import print_function

import io
import collections
import operator
import pandas as pd
import csv
import random
import os
import re

import youtokentome

import rutokenizer


input_dir = '../../data'
output_dir = '../../tmp'
tmp_dir = '../../tmp'


def is_cyr(s):
    return re.match('^[▁абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789,-]+$', s) is not None


tokenizer = rutokenizer.Tokenizer()
tokenizer.load()

# Подготовим список фраз для пакетного прогона через тренированные модели
# и последующего визуального отбора негативных примеров.
batch_phrases = set()
with io.open(os.path.join(input_dir, 'intents_batch_query.txt'), 'r', encoding='utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        if line:
            batch_phrases.add(line.strip(line))

with io.open(os.path.join(input_dir, 'test', 'test_phrases.txt'), 'r', encoding='utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        if line and not line.startswith('#'):
            batch_phrases.add(line.strip(line))

with io.open(os.path.join(input_dir, 'test', 'test_dialogues.txt'), 'r', encoding='utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        if line and not line.startswith('#') and '~' not in line and line not in ('BH', 'HBH'):
            line = line.replace('B:', '').replace('H:', '').strip()
            for s in line.split('|'):
                batch_phrases.add(s.strip())

with io.open(os.path.join(output_dir, 'intents_batch_query.amalgamation.txt'), 'w', encoding='utf-8') as wrt:
    for s in sorted(batch_phrases):
        if s:
            wrt.write('{}\n'.format(s))


input_path = os.path.join(input_dir, 'intents.txt')
output_path = os.path.join(output_dir, 'intent_dataset.csv')
samples = set()

with io.open(input_path, 'r', encoding='utf-8') as rdr:
    current_intent = None
    for iline, line in enumerate(rdr):
        if line.startswith('#'):
            if line.startswith('##'):
                if 'intent:' in line:
                    current_intent = line.split(':')[1].strip()
                else:
                    raise RuntimeError()
            else:
                # комментарии пропускаем
                continue
        else:
            line = line.strip()
            if line.startswith('-'):  # в файлах RASA строки начинаются с -
                line = line[1:]
            if line:
                if current_intent:

                    phrase = line
                    for delim in u'?,!«»"()':
                        phrase = phrase.replace(delim, ' ' + delim + ' ')

                    if phrase[-1] == '.':
                        phrase = phrase[:-1]
                    phrase = phrase.replace('  ', ' ').strip()

                    samples.add((phrase, current_intent))
                else:
                    print('line #{}: Current intent is "None"!'.format(iline))
                    exit(0)

samples = list(samples)

print('Intent frequencies:')
intent2freq = collections.Counter()
intent2freq.update(map(operator.itemgetter(1), samples))
for intent, freq in intent2freq.most_common():
    print(u'{}\t{}'.format(intent, freq))

print('TOPIC: {} samples, {} intents in result dataset ==> "{}"'.format(len(samples), len(intent2freq), output_path))
df = pd.DataFrame(columns='phrase intent'.split(), index=None)
for sample in samples:
    df = df.append({'phrase': sample[0], 'label': sample[1]}, ignore_index=True)

df.to_csv(output_path, sep='\t', header=True, index=False,
          encoding='utf-8', compression=None, quoting=csv.QUOTE_NONE, quotechar=None,
          line_terminator='\n', chunksize=None, date_format=None,
          doublequote=True,
          escapechar=None, decimal='.')

# ================ ABRACADABRA ==================
output_path = os.path.join(output_dir, 'abracadabra_dataset.csv')
samples = set()

label1 = 'абракадабра'
label0 = '0_без_абракадабры'


# Найденная в реальных диалогах абракадабра
for p in ['asr_nonsense.txt', 'gpt_nonsense.txt']: # 'invalid_syntax_dataset.txt'
    with io.open(os.path.join(input_dir, p), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s and not s.startswith('#') and '\t' not in s:
                samples.add((s, label1))

# Хорошие фразы
good_phrases = set()
for fn in ['valid_syntax_dataset.txt', 'intents_abracadabra_0.txt', 'intents.txt', 'intents_abusive_1.txt', 'intents_abusive_0.txt']:
    with io.open(os.path.join(input_dir, fn), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s and not s.startswith('#') and '\t' not in s:
                samples.add((s, label0))
                good_phrases.add(s)

                # отбелим токенизацию и тоже добавим получившийся текст как валидный
                ws = ' '.join(tokenizer.tokenize(s)).lower()
                samples.add((ws, label0))

with io.open(os.path.join(tmp_dir, 'pqa_all.dat'), 'r') as rdr:
    lines = []
    for line in rdr:
        line = line.strip()
        if line:
            lines.append(line)
        else:
            for s in lines:
                samples.add((s, label0))
                good_phrases.add(s)

            lines = []

with io.open(os.path.join(input_dir, 'moderation/детские вопросы.txt'), 'r', encoding='utf-8') as rdr:
    for line in rdr:
        s = line.strip()
        if s and not s.startswith('#') and '\t' not in s:
            samples.add((s, label0))
            good_phrases.add(s)

# Генерируем абракадабру
abrac_path = os.path.join(tmp_dir, 'good_phrases.txt')
with io.open(abrac_path, 'w', encoding='utf-8') as wrt:
    for s in good_phrases:
        wrt.write(s.lower()+'\n')

bpe_path = os.path.join(tmp_dir, 'abracadabra_yttm.model')
youtokentome.BPE.train(data=abrac_path, vocab_size=5000, model=bpe_path)
bpe = youtokentome.BPE(model=bpe_path)
vocab = list(filter(lambda s: is_cyr(s), bpe.vocab()))

# нагенерим абракадабры
for _ in range(len(good_phrases)*2):
    tx = []
    for _ in range(random.randint(5, 30)):
        t = random.choice(vocab).replace('▁', ' ')
        tx.append(t)
    s = ''.join(tx)
    if s not in good_phrases:
        samples.add((s, label1))


n1 = sum((z[1] == label1) for z in samples)
print('ABRACADABRA: {} samples, n1={} ==> "{}"'.format(len(samples), n1, output_path))
with io.open(output_path, 'w', encoding='utf-8') as wrt:
    wrt.write('phrase\tlabel\n')
    for sample in samples:
        wrt.write('{}\t{}\n'.format(sample[0], sample[1]))


# =============== ABUSIVE ===============

input1_path = os.path.join(input_dir, 'intents_abusive_1.txt')
input0_path = os.path.join(input_dir, 'intents_abusive_0.txt')
output_path = os.path.join(output_dir, 'abusive_dataset.csv')
samples = set()

label1 = 'оскорбления'
label0 = '0_без_оскорблений'

for p, label in [(input1_path, label1), (input0_path, label0)]:
    with io.open(p, 'r', encoding='utf-8') as rdr:
        for line in rdr:
            if line.startswith('#'):
                # комментарии пропускаем
                continue
            else:
                line = line.strip()
                if line:
                    samples.add((line, label))

# В качестве не-оскорбительной лексики добавим просто все фразы из датасетов PQA, там обычно нет мата и т.д.
samples0 = set()
with io.open('../../tmp/pqa_all.dat', 'r', encoding='utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        if line:
            samples0.add(line)

# Добавляем сэмплы с метками __label__OBSCENITY, __label__INSULT и __label__NORMAL
# из датасета https://www.kaggle.com/alexandersemiletov/toxic-russian-comments
with io.open('../../data/toxic_comments/alexandersemiletov.toxic_comments.dataset.txt/dataset.txt', 'r', encoding='utf-8') as rdr:
    for line in rdr:
        i = line.index(' ')
        labels = line[:i]
        text = line[i+1:].strip()
        label = None
        if '__label__NORMAL' in labels:
            label = label0
        elif '__label__INSULT' in labels or '__label__OBSCENITY' in labels:
            label = label1
        if label:
            samples.add((text, label))

n1 = sum((z[1] == label1) for z in samples)

# Ограничим кол-во негативных сэмплов, иначе будет большой дисбаланс
samples0 = sorted(list(samples0), key=lambda z: random.random())[:n1]
samples.update((s, label0) for s in samples0)

print('ABUSIVE: {} samples, n1={} ==> "{}"'.format(len(samples), n1, output_path))
df = pd.DataFrame(columns='phrase label'.split(), index=None)
for sample in samples:
    df = df.append({'phrase': sample[0], 'label': sample[1]}, ignore_index=True)

df.to_csv(output_path, sep='\t', header=True, index=False,
          encoding='utf-8', compression=None, quoting=csv.QUOTE_NONE, quotechar=None,
          line_terminator='\n', chunksize=None, date_format=None,
          doublequote=True,
          escapechar=None, decimal='.')


# ===================== SENTIMENT ======================

input_pos_path = os.path.join(input_dir, 'intents_sentiment_positive.txt')
input_neg_path = os.path.join(input_dir, 'intents_sentiment_negative.txt')
input_zero_path = os.path.join(input_dir, 'intents_sentiment_neutral.txt')
output_path = os.path.join(output_dir, 'sentiment_dataset.csv')
samples = set()

label_posit = 'выражение_одобрения'
label_negat = 'выражение_неодобрения'
label_zero = '0_нейтральный_сентимент'

for p, label in [(input_pos_path, label_posit), (input_neg_path, label_negat), (input_zero_path, label_zero)]:
    with io.open(p, 'r', encoding='utf-8') as rdr:
        for line in rdr:
            if line.startswith('#'):
                # комментарии пропускаем
                continue

            else:
                line = line.strip()
                if line:
                    samples.add((line, label))

df = pd.read_csv('/media/inkoziev/corpora/Corpus/snalp/sdict-11545.csv', delimiter=';')
df = df.head(n=1000)
df = df[df.score < 0.0]
samples.update((text, label_negat) for text in df.word)

# отзывы по отелям
with io.open('/media/inkoziev/corpora/Corpus/sentiment/polarity_corpus.dat', 'r', encoding='utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        fields = line.split('\t')
        if len(fields) == 2:
            text = fields[0]
            if fields[1] == 'posit=1':
                samples.add((text, label_posit))
            elif fields[1] == 'negat=1':
                samples.add((text, label_negat))

# rusentiment
df_r = pd.read_csv('../../data/rusentiment_datasets/rusentiment_preselected_posts.csv', delimiter=',')
for i, r in df_r.iterrows():
    label = r['label']
    text = r['text']
    if '\n' not in text:
        if label == 'negative':
            samples.add((text, label_negat))
        elif label == 'neutral':
            samples.add((text, label_zero))
        elif label == 'positive':
            samples.add((text, label_posit))


n_neg = sum((s[1] == label_negat) for s in samples)
n_pos = sum((s[1] == label_posit) for s in samples)
n_zero = sum((s[1] == label_zero) for s in samples)

print('SENTIMENT: {} samples n_neg={} n_pos={} n_zero={} ==> "{}"'.format(len(samples), n_neg, n_pos, n_zero, output_path))
df = pd.DataFrame(columns='phrase label'.split(), index=None)
for sample in samples:
    df = df.append({'phrase': sample[0], 'label': sample[1]}, ignore_index=True)

df.to_csv(output_path, sep='\t', header=True, index=False,
          encoding='utf-8', compression=None, quoting=csv.QUOTE_NONE, quotechar=None,
          line_terminator='\n', chunksize=None, date_format=None,
          doublequote=True,
          escapechar=None, decimal='.')


# ============================ НАПРАВЛЕННОСТЬ ФРАЗЫ ============================

input_bot_path = os.path.join(input_dir, 'intents_dir_bot.txt')
input_interrog_path = os.path.join(input_dir, 'intents_dir_interrogator.txt')
input_other_path = os.path.join(input_dir, 'intents_dir_other.txt')
output_path = os.path.join(output_dir, 'direction_dataset.csv')
samples = set()

label_bot = 'прочие_вопросы_к_боту'
label_interrog = 'вопросы_к_собеседнику'
label_other = '0_вопросы_о_ком_то_еще'

for p, label in [(input_bot_path, label_bot), (input_interrog_path, label_interrog), (input_other_path, label_other)]:
    with io.open(p, 'r', encoding='utf-8') as rdr:
        for line in rdr:
            if line.startswith('#'):
                # комментарии пропускаем
                continue

            else:
                line = line.strip()
                if line:
                    samples.add((line, label))

print('DIRECTION: {} samples ==> "{}"'.format(len(samples), output_path))
df = pd.DataFrame(columns='phrase label'.split(), index=None)
for sample in samples:
    df = df.append({'phrase': sample[0], 'label': sample[1]}, ignore_index=True)

df.to_csv(output_path, sep='\t', header=True, index=False,
          encoding='utf-8', compression=None, quoting=csv.QUOTE_NONE, quotechar=None,
          line_terminator='\n', chunksize=None, date_format=None,
          doublequote=True,
          escapechar=None, decimal='.')


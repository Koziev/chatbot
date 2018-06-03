# -*- coding: utf-8 -*-
'''
Поиск наиболее подходящей предпосылки (предложения-факта) из списка (базы знаний)
с использованием предварительно натренированной модели.

С консоли вводится вопрос, предпосылки читаются из текстового файла. После расчета по модели
в консоль выводится несколько самых релевантных фактов.

Тренировка модели - см. скрипты train_xgb_relevancy.sh или train_lgb_relevancy.sh
в папке scripts. 
'''

from __future__ import division  # for python2 compatability
from __future__ import print_function

import codecs
import os
import sys
import argparse

from bot.text_utils import TextUtils
from bot.xgb_relevancy_detector import XGB_RelevancyDetector
from bot.lgb_relevancy_detector import LGB_RelevancyDetector


def sanitize(phrase, text_utils):
    return u' '.join(text_utils.tokenize(phrase)).lower()


parser = argparse.ArgumentParser(description='Relevant fact finder')
parser.add_argument('--facts_path', type=str, default='../data/premises.txt', help='path to text file with facts')
parser.add_argument('--model_dir', type=str, default='../tmp', help='folder with trained model datafiles')
parser.add_argument('--model', type=str, default='lgb', help='model to use: xgb | lgb')

args = parser.parse_args()
facts_path = args.facts_path
model_dir = args.model_dir
model = args.model

# реализация операций с текстом на целевом языке (токенизация и т.д.) скрыта в классе TextUtils.
text_utils = TextUtils()

relevancy_detector = None
if model == 'xgb':
    relevancy_detector = XGB_RelevancyDetector()
elif model == 'lgb':
    relevancy_detector = LGB_RelevancyDetector()
else:
    print('Unknown model {}'.format(model))
    exit(1)

relevancy_detector.load(model_dir)

# Загружаем базу знаний. Каждая строка в этом файле рассматривается как предложение-факт.
print('Loading facts from {}...'.format(facts_path))
memory_phrases = []
with codecs.open(facts_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        premise = line.strip()
        if len(premise)>0:
            premise = sanitize(premise, text_utils)
            memory_phrases.append((premise, None, None))

nb_answers = len(memory_phrases)
print('{} facts in knowledge base'.format(nb_answers))

trace_enabled = False

while True:
    question = raw_input(':> ').decode(sys.stdout.encoding).strip().lower()
    if len(question) == 0:  # пустой вопрос - выход
        break

    question = sanitize(question, text_utils)

    best_phrases, best_rels = relevancy_detector.get_most_relevant( question,
                                                                    memory_phrases,
                                                                    text_utils,
                                                                    word_embeddings=None,
                                                                    nb_results=nb_answers)

    # выведем 10 самых релевантных фактов для введенного вопроса в порядке убывания
    # релевантности.
    print('\n')
    for answer, sim in zip(best_phrases, best_rels)[0:10]:
        print(u'{:<8.4f} {}'.format(sim, answer))
    print('\n\n')

print('Bye...')

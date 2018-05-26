# -*- coding: utf-8 -*-
"""
Оценка качества необучаемой метрики (космнусная близость векторов из doc2vec модели)
на задаче выбора релевантного факта для вопроса.
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import os
import codecs
import sys

from utils.tokenizer import Tokenizer
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup

doc2vec_path = '../tmp/doc2vec.txt'
data_folder = '../data'


def v_cosine( a, b ):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


tokenizer = Tokenizer()

eval_data = EvaluationDataset(0, tokenizer)
eval_data.load(data_folder)

# Нам будут нужны doc2vec векторы всех предложений, участвующих в вычислениях
# похожести. Получим список этих предложений и загрузим их из doc2vec.txt.
all_phrases = eval_data.get_all_phrases()
phrase2vec = dict()
with codecs.open(doc2vec_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        tx = line.strip().split(u'\t')
        if len(tx) == 2:
            phrase = tx[0]
            if phrase == u'мышка любит сыр':
                pass
            if phrase in all_phrases:
                v = np.fromstring(tx[1], sep=u' ')
                phrase2vec[phrase] = v

nb_good = 0
nb_bad = 0

for irecord, phrases in eval_data.generate_groups():
    y_pred = []
    for irow, (premise_words, question_words) in enumerate(phrases):
        premise = u' '.join(premise_words)
        premise_v = phrase2vec[premise] if premise in phrase2vec else None

        question = u' '.join(question_words)
        question_v = phrase2vec[question] if question in phrase2vec else None

        if premise_v is not None and question_v is not None:
            sim = v_cosine(premise_v, question_v)
        else:
            sim = 0.0
        y_pred.append(sim)

    # предпосылка с максимальной релевантностью
    max_index = np.argmax(y_pred)
    selected_premise = u' '.join(phrases[max_index][0]).strip()

    # эта выбранная предпосылка соответствует одному из вариантов
    # релевантных предпосылок в этой группе?
    if eval_data.is_relevant_premise(irecord, selected_premise):
        nb_good += 1
        print(EvaluationMarkup.ok_color + EvaluationMarkup.ok_bullet + EvaluationMarkup.close_color, end='')
    else:
        nb_bad += 1
        print(EvaluationMarkup.fail_color + EvaluationMarkup.fail_bullet + EvaluationMarkup.close_color, end='')

    max_sim = np.max(y_pred)

    question_words = phrases[0][1]
    print(u'{:<40} {:<40} {}/{}'.format(u' '.join(question_words), u' '.join(phrases[max_index][0]), y_pred[max_index],
                                        y_pred[0]))

# Итоговая точность выбора предпосылки.
accuracy = float(nb_good) / float(nb_good + nb_bad)
print('accuracy={}'.format(accuracy))

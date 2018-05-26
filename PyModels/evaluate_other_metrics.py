# -*- coding: utf-8 -*-
"""
Оценка качества выбора наиболее релевантной предпосылки для не-обучаемых метрик типа WMD.
"""

from __future__ import division
from __future__ import print_function

import codecs
import itertools
import sys
import argparse
import math

import gensim
import numpy as np
import scipy.spatial.distance

from utils.tokenizer import Tokenizer
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup


def enclose_phrase(phrase):
    return u'\b ' + phrase.strip() + u' \n'


def ngrams(s, n):
    return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1&shingles2))/float(len(shingles1|shingles2))


def get_average_vector(words, w2v):
    v = None
    denom = 0
    for iword, word in enumerate(words):
        if word in w2v:
            denom += 1
            if v is None:
                v = np.array(w2v[word])
            else:
                v += w2v[word]

    return v/denom


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


parser = argparse.ArgumentParser(description='Non-trainable metrics evaluation on relevancy dataset')
parser.add_argument('--metric', type=str, default='jaccard')
parser.add_argument('--shingle_len', type=int, default=3, help='length of character shingles')
parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')
parser.add_argument('--word2vector', type=str, default='/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=32.bin', help='path to word2vector model dataset')

args = parser.parse_args()

data_folder = args.data_dir

metrics = ['jaccard', 'wmd', 'w2v']
metric = args.metric
if metric not in metrics:
    print('Unknown metric {}, use one of {}'.format(metric, ' '.join(metrics)))
    exit(1)

if metric == 'jaccard':
    shingle_len = args.shingle_len

if metric in ['wmd', 'w2v']:
    word2vector_path = args.word2vector
    print( 'Loading the w2v model {}'.format(word2vector_path) )
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=True)
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))


tokenizer = Tokenizer()

eval_data = EvaluationDataset(0, tokenizer)
eval_data.load(data_folder)

nb_good = 0
nb_bad = 0

for irecord, phrases in eval_data.generate_groups():
    y_pred = []
    for irow, (premise_words, question_words) in enumerate(phrases):
        sim = -1
        if metric == 'jaccard':
            sim = jaccard(enclose_phrase(u' '.join(premise_words)), enclose_phrase(u' '.join(question_words)), shingle_len)
        elif metric == 'wmd':
            wmd = w2v.wmdistance(premise_words, question_words)
            sim = math.exp(-wmd)
        elif metric == 'w2v':
            v1 = get_average_vector(premise_words, w2v)
            v2 = get_average_vector(question_words, w2v)
            sim = v_cosine(v1, v2)

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

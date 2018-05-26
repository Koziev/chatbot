# -*- coding: utf-8 -*-
"""
Программа для поиска наиболее релевантных предложений в корпусе для введенного
в консоли предложения. Используется для оценки качества моделей релевантности
в проекте чат-бота https://github.com/Koziev/chatbot

Метрика и прочие параметры задаются опциями командной строки при запуске.
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

# -------------------------------------------------------------------

def enclose_phrase(phrase):
    return u'\b ' + phrase.strip() + u' \n'


def ngrams(s, n):
    return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    #return distance.jaccard(shingles1, shingles2)
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


parser = argparse.ArgumentParser(description='Neural model for text relevance estimation')
parser.add_argument('--facts', type=str, default='../tmp/premises.txt', help='path to corpus file containing sentences')
parser.add_argument('--max_nb_facts', type=int, default=1000000, help='max number of sentences to read from corpus')
parser.add_argument('--metric', type=str, default='jaccard')
parser.add_argument('--word2vector', type=str, default='f:/Word2Vec/word_vectors_cbow=1_win=5_dim=32.txt', help='path to word2vector model dataset')
parser.add_argument('--shingle_len', type=int, default=3, help='length of character shingles')

args = parser.parse_args()

facts_path = args.facts
max_nb_facts = args.max_nb_facts

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
    w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=False)
    w2v_dims = len(w2v.syn0[0])
    print('w2v_dims={0}'.format(w2v_dims))


tokenizer = Tokenizer()
phrases1 = []
with codecs.open(facts_path, 'r', 'utf-8') as rdr:
    for phrase in rdr:
        words = tokenizer.tokenize(phrase.strip())
        if len(words) > 0:
            if metric == 'w2v':
                phrases1.append((phrase.strip(), words, get_average_vector(words, w2v)))
            else:
                phrases1.append((phrase.strip(), words))
            if len(phrases1) >= max_nb_facts:
                break

nb_phrases = len(phrases1)
print(u'{1} phrases are loaded from {0}'.format(facts_path, nb_phrases))

while True:
    phrase2 = raw_input('phrase #2:> ').decode(sys.stdout.encoding).strip().lower()
    if len(phrase2) == 0:
        break

    words2 = tokenizer.tokenize(phrase2)

    phrase_sims = []

    if metric == 'jaccard':
        for phrase1 in phrases1:
            sim = jaccard(enclose_phrase(phrase1[0]), enclose_phrase(phrase2), shingle_len)
            phrase_sims.append((phrase1, sim))
    elif metric == 'wmd':
        for phrase1 in phrases1:
            words1 = phrase1[1]
            wmd = w2v.wmdistance(words1, words2)
            sim = math.exp(-wmd)
            phrase_sims.append((phrase1, sim))
    elif metric == 'w2v':
        v2 = get_average_vector(words2, w2v)
        for phrase1 in phrases1:
            v1 = phrase1[2]
            #sim = scipy.spatial.distance.cosine(v1, v2)
            sim = v_cosine(v1, v2)
            phrase_sims.append((phrase1, sim))

    phrase_sims = sorted(phrase_sims, key=lambda z: -z[1])

    # Выведем top N фраз
    for phrase1, rel in phrase_sims[0:10]:
        print(u'{:4f}\t{}'.format(rel, phrase1[0]))


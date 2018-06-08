# -*- coding: utf-8 -*-
"""
Эксперимент: генерация SDR слов через нелинейное понижение размерности
1-hot кодов символов средствами scikit learn.
"""

from __future__ import division
from __future__ import print_function

import gc
import itertools
import json
import os
import sys
import argparse
import codecs
import six

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer


wordlist_path = '../tmp/dataset_words.txt'
result_path = '../tmp/wordchar2sdr.txt'

words = list(filter(lambda w: len(w) > 0, [line.strip() for line in codecs.open(wordlist_path, 'r', 'utf-8')]))
nb_words = len(words)
print('{} words'.format(nb_words))

all_chars = set(itertools.chain(*words))
nb_chars = len(all_chars)
char2index = dict((c, i) for (i, c) in enumerate(itertools.chain(u' ', filter(lambda c: c!=u' ', all_chars))))
max_word_len = max(map(len, words))

old_dim = nb_chars*max_word_len
X = np.zeros((nb_words, old_dim), dtype=bool)
for iword, word in enumerate(words):
    for ichar, c in enumerate(word):
        X[iword, ichar*nb_chars + char2index[c]] = True

new_dim = 256
print('Reducing old_dim={} new_dim={}'.format(old_dim, new_dim))
sdr_generator = None
sdr_generator = SparseRandomProjection(n_components=new_dim, density=0.01, eps=0.1)

X2 = sdr_generator.fit_transform(X)

n10 = X2.size
n0 = sum(np.reshape(X2 == 0.0, (X2.size)))
n1 = sum(np.reshape(X2 != 0.0, (X2.size)))
density = float(n1) / float(n10)
print('n0={} n1={} n10={} density={}'.format(n0, n1, n10, density))

print('Storing wordchar vectors to {}'.format(result_path))
with codecs.open(result_path, 'w', 'utf-8') as wrt:
    wrt.write('{} {}\n'.format(nb_words, new_dim))

    for word, word_vect in itertools.izip(words, X2):
        wrt.write(u'{} {}\n'.format(word, u' '.join([str(int(x != 0)) for x in word_vect])))

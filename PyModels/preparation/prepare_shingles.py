# -*- coding: utf-8 -*-
'''
Подготовка базы со статистикой символьных шинглов для генератора ответа в чат-боте.
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import codecs
import itertools
import os
import collections

from utils.tokenizer import Tokenizer


data_folder = '../../data'
infiles = ['facts4_1s.txt', 'facts4_2s.txt', 'facts4.txt',
           'facts5_1s.txt', 'facts5_2s.txt', 'facts5.txt',
           'facts6_1s.txt', 'facts6_2s.txt', 'facts6.txt',
           'facts7_1s.txt', 'facts7_2s.txt', 'facts7.txt',]

respath = '../../data/shingles.tsv'

SHINGLE_LEN = 3
MIN_FREQ = 20

def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    """
    Цепочку слов соединяем в строку, добавляя перед цепочкой и после нее
    пробел и специальные символы начала и конца.
    :param words:
    :return:
    """
    return u' ' + u' '.join(words) + u' '


tokenizer = Tokenizer()
ngram2count = collections.Counter()

for inpath in infiles:
    with codecs.open(os.path.join(data_folder, inpath), 'r', 'utf8') as rdr:
        for line in rdr:
            line = line.strip()
            phrase = words2str(tokenizer.tokenize(line))
            ngram2count.update(ngrams(phrase, 3))

print('Total number of shingles={}'.format(len(ngram2count)))

with codecs.open(respath, 'w', 'utf-8') as wrt:
    for shingle, freq in ngram2count.most_common():
        if freq >= MIN_FREQ:
            wrt.write(u'{}\t{}\n'.format(shingle, freq))

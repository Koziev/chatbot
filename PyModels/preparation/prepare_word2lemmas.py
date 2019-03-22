# -*- coding: utf-8 -*-
"""
Подготовка файла со списком грамматических форм.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import io
import gzip

inpath = '../../data/word2lemma.dat'
outpath = '../../data/ru_word2lemma.tsv.gz'

dataset_words = set()
with io.open('../../tmp/dataset_words.txt', 'r', encoding='utf-8') as rdr:
    for line in rdr:
        dataset_words.add(line.strip())

print(u'Storing data to {}'.format(outpath))
with io.open(inpath, 'r', encoding='utf-8') as rdr:
    with gzip.open(outpath, 'w') as wrt:
        for line in rdr:
            if u' ' not in line:
                tx = line.strip().split(u'\t')
                if len(tx) == 3:
                    word = tx[0].lower()
                    if word in dataset_words:
                        lemma = tx[1].lower()
                        wrt.write(u'{}\t{}\n'.format(word, lemma).encode('utf8'))

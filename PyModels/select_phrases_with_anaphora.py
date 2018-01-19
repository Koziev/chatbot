# -*- coding: utf-8 -*-
'''
'''

from __future__ import print_function
from gensim.models import word2vec
import logging
import os
import random
from collections import Counter
import codecs
from Segmenter import  Segmenter
from Tokenizer import Tokenizer
import collections
import tqdm


# путь к файлу с исходным текстовым корпусом.
corpus_paths = [ r'f:\Corpus\SENTx\ru\SENT4.txt',
                 r'f:\Corpus\SENTx\ru\SENT5.txt',
                 r'f:\Corpus\SENTx\ru\SENT6.txt' ]

pronouns = set(u'нее него них ней ним ними нему нами'.split())

tokenizer = Tokenizer()

for corpus_path in corpus_paths:
    print('Counting lines in {}'.format(corpus_path))
    total_lines = 0
    with open(corpus_path) as f:
        for i, l in enumerate(f):
            pass
    total_lines = i + 1
    print('total_lines={}'.format(total_lines))

    print('Selecting phrases from {}'.format(corpus_path))
    with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
        with codecs.open('f:/tmp/anaphora.txt', 'w', 'utf-8') as wrt:
            for sent in tqdm.tqdm(rdr, total=total_lines, desc='Selecting'):
                words = tokenizer.tokenize_raw(sent)
                ok = False
                for word in words:
                    if word.lower() in pronouns:
                        ok = True
                        break

                if ok:
                    wrt.write(u'{}\n'.format(sent))

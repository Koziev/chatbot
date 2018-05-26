# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import os
import codecs
import sys
import gensim
from gensim.models.doc2vec import TaggedDocument

from utils.tokenizer import Tokenizer

doc2vec_path = '../tmp/doc2vec.txt'


def v_cosine( a, b ):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

tokenizer = Tokenizer()

while True:
    s = raw_input('phrase: ').decode(sys.stdout.encoding).strip().lower()
    phrase1 = u' '.join(tokenizer.tokenize(s))
    v1 = None
    with codecs.open(doc2vec_path, 'r', 'utf-8') as rdr:
        for line in rdr:
            tx = line.strip().split(u'\t')
            if len(tx) == 2:
                phrase = tx[0]
                if phrase == phrase1:
                    v1 = np.fromstring(tx[1], sep=u' ')
                    break

    if v1 is None:
        print('Phrase not found in doc2vec dictionary')
    else:
        sim_max = -np.inf
        sim_min = -np.inf
        hits = []
        with codecs.open(doc2vec_path, 'r', 'utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split(u'\t')
                if len(tx) == 2:
                    phrase2 = tx[0]
                    if phrase2 != phrase1:
                        v2 = np.fromstring(tx[1], sep=u' ')
                        sim = v_cosine(v1, v2)
                        if sim >= sim_min:
                            hits.append((sim, phrase2))

                            if len(hits) > 10:
                                hits = sorted(hits, key=lambda z: -z[0])[:10]
                                sim_max = max(z[0] for z in hits)
                                sim_min = min(z[0] for z in hits)

        print('Most similar phrases:')
        for sim, phrase in hits:
            print(u'{:<50s} {}'.format(phrase, sim))





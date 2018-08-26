# -*- coding: utf-8 -*-

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import itertools
import logging
import json
import os
import sys
import gzip


class Word2Lemmas(object):
    def __init__(self):
        pass

    def load(self, path):
        logging.info(u'Loading lexicon from {}'.format(path))
        self.lemmas = dict()
        self.forms = dict()
        with gzip.open(path, 'r') as rdr:
            for line in rdr:
                tx = line.strip().decode('utf8').split('\t')
                if len(tx) == 2:
                    form = tx[0]
                    lemma = tx[1]

                    if form not in self.forms:
                        self.forms[form] = [lemma]
                    else:
                        self.forms[form].append(lemma)

                    if lemma not in self.lemmas:
                        self.lemmas[lemma] = {form}
                    else:
                        self.lemmas[lemma].add(form)
        logging.info('Lexicon loaded: {} lemmas, {} wordforms'.format(len(self.lemmas), len(self.forms)))

    def get_forms(self, word):
        if word in self.forms:
            #result = set()
            #for lemma in self.forms[word]:
            #    result.update(self.lemmas[lemma])
            #return result
            return set(itertools.chain(*(self.lemmas[lemma] for lemma in self.forms[word])))
        else:
            return [word]

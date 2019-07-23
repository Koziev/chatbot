# -*- coding: utf-8 -*-

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import itertools
import logging
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

        self.lemmas[u'ты'] = u'ты тебя тебе тобой'.split()
        self.lemmas[u'я'] = u'я меня мной мне'.split()
        logging.info('Lexicon loaded: {} lemmas, {} wordforms'.format(len(self.lemmas), len(self.forms)))

    def has_forms(self, lemma):
        return lemma in self.lemmas

    def get_lemma_forms(self, lemma):
        """Возвращает все варианты форм слова, заданного леммой lemma"""
        if lemma in self.lemmas:
            return self.lemmas[lemma]
        else:
            return [lemma]

    def get_forms(self, word):
        if word in self.forms:
            return set(itertools.chain(*(self.lemmas[lemma] for lemma in self.forms[word])))
        else:
            return [word]

# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import codecs
import itertools
import os
import sys

from utils.tokenizer import Tokenizer
from utils.padding_utils import pad_wordseq


class EvaluationGroup(object):
    """
    Хранилище группы вопросов и релевантных для них предпосылок.
    """
    def __init__(self, max_wordseq_len, tokenizer):
        self.max_wordseq_len = max_wordseq_len
        self.tokenizer = tokenizer
        self.premises_str = []
        self.premises = []  # списки слов
        self.questions = []  # списки слов

    def is_empty(self):
        return len(self.premises) == 0

    def invalid_format(self):
        raise Exception('Invalid format of evaluation dataset file')

    def load(self, rdr):
        for line in rdr:
            if len(line) == 0:
                eof_reached = True
                break

            line = line.strip()
            if len(line) == 0:
                if len(self.premises) > 0:
                    break
                else:
                    continue

            if line.startswith(u'T:'):
                if len(self.questions) > 0:
                    self.invalid_format()

                premise = line.replace(u'T:', u'').replace(u'ё', u'е').lower().strip()
                premise_words = pad_wordseq(self.tokenizer.tokenize(premise), self.max_wordseq_len)
                self.premises_str.append(u' '.join(premise_words))
                self.premises.append(premise_words)
            elif line.startswith(u'Q:'):
                question = line.replace(u'Q:', u'').replace(u'ё', u'е').strip()
                question = pad_wordseq(self.tokenizer.tokenize(question), self.max_wordseq_len)
                self.questions.append(question)
            else:
                self.invalid_format()

        eof_reached = len(self.premises) == 0
        return eof_reached

    def is_relevant_premise(self, premise_str):
        return premise_str in self.premises_str


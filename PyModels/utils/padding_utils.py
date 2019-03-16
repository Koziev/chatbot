# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import itertools

PAD_WORD = u''


def lpad_wordseq(words, n):
    if len(words) >= n:
        return words
    else:
        return list(itertools.chain(itertools.repeat(PAD_WORD, n - len(words)), words))


def rpad_wordseq(words, n):
    if len(words) >= n:
        return words
    else:
        return list(itertools.chain(words, itertools.repeat(PAD_WORD, n - len(words))))

# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import itertools

PAD_WORD = u''


def pad_wordseq(words, n):
    if len(words) >= 0:
        return words
    else:
        return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))

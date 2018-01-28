# -*- coding: utf-8 -*-

import itertools
import re
from utils.tokenizer import Tokenizer


BEG_WORD = u'\b'
END_WORD = u'\n'
PAD_WORD = u''

class TextUtils(object):
    def __init__(self):
        self.tokenizer = Tokenizer()

    def canonize_text(self, s):
        # Удаляем два и более пробелов подряд, заменяя на один.
        s = re.sub("(\\s{2,})", ' ', s.strip())
        return s

    def ngrams(self, s, n):
        return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]

    def words2str(self, words):
        return u' '.join(itertools.chain([BEG_WORD], filter(lambda z: len(z) > 0, words), [END_WORD]))

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    # Слева добавляем пустые слова
    def pad_wordseq(self, words, n):
        return list( itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words ) )

    # Справа добавляем пустые слова
    def rpad_wordseq(self, words, n):
        return list( itertools.chain( words, itertools.repeat(PAD_WORD, n-len(words)) ) )


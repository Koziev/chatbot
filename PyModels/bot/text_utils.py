# -*- coding: utf-8 -*-
"""
Код для выполнения операций с текстом на русском языке (или другом целевом),
в частности - токенизация, лемматизация.
"""

import itertools
import re
import os

from pymystem3 import Mystem
from utils.tokenizer import Tokenizer
from word2lemmas import Word2Lemmas
from language_resources import LanguageResources


BEG_WORD = u'\b'
END_WORD = u'\n'
PAD_WORD = u''


class TextUtils(object):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.lemmatizer = Mystem()
        self.lexicon = Word2Lemmas()
        self.language_resources = LanguageResources();

    def load_dictionaries(self, data_folder):
        word2lemmas_path = os.path.join(data_folder, 'ru_word2lemma.tsv.gz')
        self.lexicon.load(word2lemmas_path)

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

    def lemmatize(self, s):
        words = self.tokenizer.tokenize(s)
        wx = u' '.join(words)
        return [l for l in self.lemmatizer.lemmatize(wx) if len(l.strip()) > 0]

    # Слева добавляем пустые слова
    def lpad_wordseq(self, words, n):
        return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))

    # Справа добавляем пустые слова
    def rpad_wordseq(self, words, n):
        return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))

    def get_lexicon(self):
        return self.lexicon

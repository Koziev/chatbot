# -*- coding: utf-8 -*-
"""
Код для выполнения операций с текстом на русском языке (или другом целевом),
в частности - токенизация, лемматизация, частеречная разметка.
"""

import itertools
import re
import os

from pymystem3 import Mystem
import rupostagger
from utils.tokenizer import Tokenizer
from word2lemmas import Word2Lemmas
from language_resources import LanguageResources


BEG_WORD = u'\b'
END_WORD = u'\n'
PAD_WORD = u''


class TextUtils(object):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.load()
        self.lemmatizer = Mystem()
        self.lexicon = Word2Lemmas()
        self.language_resources = LanguageResources()
        self.postagger = rupostagger.RuPosTagger()

    def load_dictionaries(self, data_folder):
        word2lemmas_path = os.path.join(data_folder, 'ru_word2lemma.tsv.gz')
        self.lexicon.load(word2lemmas_path)

        word2tags_path = os.path.join(data_folder, 'chatbot_word2tags.dat')
        self.postagger.load(word2tags_path)

    def tag(self, words):
        """ Частеречная разметка для цепочки слов words """
        return self.postagger.tag(words)

    def canonize_text(self, s):
        """ Удаляем два и более пробелов подряд, заменяя на один """
        s = re.sub("(\\s{2,})", ' ', s.strip())
        return s

    def remove_terminators(self, s):
        """ Убираем финальные пунктуаторы ! ? ."""
        return s[:-1].strip() if s[-1] in u'?!.' else s

    def wordize_text(self, s):
        return u' '.join(self.tokenize(s))

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

    def lpad_wordseq(self, words, n):
        """ Слева добавляем пустые слова """
        return list(itertools.chain(itertools.repeat(PAD_WORD, n - len(words)), words))

    def rpad_wordseq(self, words, n):
        """ Справа добавляем пустые слова """
        return list(itertools.chain(words, itertools.repeat(PAD_WORD, n - len(words))))

    def get_lexicon(self):
        return self.lexicon

    def is_question_word(self, word):
        return word in u'кто что почему отууда куда зачем чего кого кем чем кому чему ком чем как сколько ли когда докуда'.split()

# -*- coding: utf-8 -*-
"""
Простой лемматизатор - разбивает строку на токены и возвращает список
словарных форм.
"""

from __future__ import print_function

from pymystem3 import Mystem


class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = Mystem()

    @staticmethod
    def is_good_token(t):
        return len(t.strip()) > 0 and not t[0] in u'.,!?-'

    def tokenize(self, phrase):
        lemmas = [lemma for lemma in self.lemmatizer.lemmatize(phrase) if Lemmatizer.is_good_token(lemma)]
        return lemmas

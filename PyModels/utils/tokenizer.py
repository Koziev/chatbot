# -*- coding: utf-8 -*-
"""
Простой токенизатор - разбивает строку на слова
"""

import rutokenizer


class Tokenizer(rutokenizer.Tokenizer):
    def tokenize(self, phrase):
        return [word.lower().replace(u'ё', u'е')
                for word
                in super(Tokenizer, self).tokenize(phrase)]

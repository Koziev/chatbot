# -*- coding: utf-8 -*-
'''
Простой токенизатор - разбивает строку на слова
'''

from __future__ import print_function
import re


class Tokenizer(object):
    def __init__(self):
        self.regex = re.compile(r'[%s\s]+' % re.escape(u'[ +<>`~; .,?？!-…№”“„{}/\'"–—_:«»*]()）》\t'))

    def normalize_word(self, w):
        i = w.find(u'°')
        if i!=-1 and i>0 and i<len(w)-1 and sum([1 for c in w if c.isdigit() ])==0:
            return w.lower()  #.replace( u'°', u'е' )
        else:
            return w.lower().replace( u'ё', u'е' )

    def tokenize(self, phrase):
        return [ self.normalize_word(w) for w in self.regex.split(phrase) if len(w)>0 ]

    def tokenize_raw(self, phrase):
        return [ w for w in self.regex.split(phrase) if len(w)>0 ]

# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import rupostagger
import rulemma

#from pymystem3 import Mystem
#from nltk.stem.snowball import RussianStemmer, EnglishStemmer
from ruchatbot.utils.tokenizer import Tokenizer



class PhraseSplitter(object):
    def __init__(self):
        pass

    def tokenize(self, phrase):
        raise NotImplementedError()

    @staticmethod
    def create_splitter(mode):
        if mode == 0:
            t = PhraseTokenizer()
        elif mode == 1:
            t = PhraseLemmatizer()
        elif mode == 2:
            t = PhraseStemmer()

        return t


class PhraseTokenizer(PhraseSplitter):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.load()

    def tokenize(self, phrase):
        return self.tokenizer.tokenize(phrase)


class PhraseLemmatizer(PhraseSplitter):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.load()
        #self.lemmatizer = Mystem()
        self.tagger = rupostagger.RuPosTagger()
        self.tagger.load()
        self.lemm = rulemma.Lemmatizer()
        self.lemm.load()

    def extract_lemma(self, token):
        return token[0] if token[1] == 'PRON' else token[2]

    def tokenize(self, phrase):
        words = self.tokenizer.tokenize(phrase)
        # вариант с pymystem
        #wx = u' '.join(words)
        #return [l for l in self.lemmatizer.lemmatize(wx) if len(l.strip()) > 0]

        # вариант с собственным лемматизатором
        tags = self.tagger.tag(words)
        tokens = self.lemm.lemmatize(tags)
        return [self.extract_lemma(t) for t in tokens]


class PhraseStemmer(PhraseSplitter):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.stemmer = RussianStemmer()

    def tokenize(self, phrase):
        return [self.stemmer.stem(w) for w in self.tokenizer.tokenize(phrase) if len(w.strip()) > 0]

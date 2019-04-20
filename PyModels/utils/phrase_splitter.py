from __future__ import division
from __future__ import print_function

from pymystem3 import Mystem
from nltk.stem.snowball import RussianStemmer, EnglishStemmer
from utils.tokenizer import Tokenizer


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
        self.lemmatizer = Mystem()

    def tokenize(self, phrase):
        words = self.tokenizer.tokenize(phrase)
        wx = u' '.join(words)
        return [l for l in self.lemmatizer.lemmatize(wx) if len(l.strip()) > 0]


class PhraseStemmer(PhraseSplitter):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.stemmer = RussianStemmer()

    def tokenize(self, phrase):
        return [self.stemmer.stem(w) for w in self.tokenizer.tokenize(phrase) if len(w.strip()) > 0]

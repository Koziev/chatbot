# -*- coding: utf-8 -*-

import collections
import random

try:
    from itertools import izip as zip
except ImportError:
    pass


def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(shingles1, shingles2):
    return float(len(shingles1 & shingles2)) / float(len(shingles1 | shingles2))


BEG_CHAR = '['
END_CHAR = ']'


class CorpusSearcher(object):
    """
    Для генерации негативных паттернов нам надо будет для каждого
    предложения быстро искать близкие к нему по критерию Жаккара.
    Экземпляры этого класса хранят список фраз и их индекс для
    быстрого подбора.
    """
    def __init__(self):
        self.phrases_set = set()
        self.phrases = []
        self.shingle2sent = dict()

    @staticmethod
    def dress(phrase):
        return BEG_CHAR + phrase + END_CHAR

    def add_phrase(self, phrase):
        if phrase not in self.phrases_set:
            self.phrases.append(phrase)
            self.phrases_set.add(phrase)
            for shingle in ngrams(CorpusSearcher.dress(phrase), 3):
                if shingle not in self.shingle2sent:
                    self.shingle2sent[shingle] = set([phrase])
                else:
                    self.shingle2sent[shingle].add(phrase)

    def find_similar(self, phrase, nb_phrases):
        phrase2hits = collections.Counter()
        for shingle in ngrams(CorpusSearcher.dress(phrase), 3):
            if shingle in self.shingle2sent:
                phrase2hits.update(self.shingle2sent[shingle])

        return [phrase2 for (phrase2, cnt) in phrase2hits.most_common(nb_phrases)]

    def get_random(self, nb_phrases):
        # return np.random.choice(self.phrases, size=nb_phrases, replace=False)
        for _ in range(nb_phrases):
            yield random.choice(self.phrases)

    def __len__(self):
        return len(self.phrases)

# -*- coding: utf-8 -*-
"""
Необучаемая реализация модели определения похожести двух фраз через
вычисление коэффициента Жаккара.
"""

import numpy as np
import logging
import itertools
from synonymy_detector import SynonymyDetector


class Jaccard_SynonymyDetector(SynonymyDetector):
    def __init__(self):
        super(Jaccard_SynonymyDetector, self).__init__()
        self.logger = logging.getLogger('Jaccard_SynonymyDetector')

    def load(self, models_folder):
        pass

    @staticmethod
    def ngrams(s, n):
        return set(u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)]))

    @staticmethod
    def jaccard(s1, s2, shingle_len):
        shingles1 = Jaccard_SynonymyDetector.ngrams(s1.lower(), shingle_len)
        shingles2 = Jaccard_SynonymyDetector.ngrams(s2.lower(), shingle_len)
        return float(len(shingles1 & shingles2)) / float(len(shingles1 | shingles2))

    def get_most_similar(self, probe_phrase, phrases, text_utils, word_embeddings, nb_results=1):
        assert(nb_results > 0)
        assert(len(probe_phrase) != 0)

        y = []
        for iphrase, phrase2 in enumerate(phrases):
            y.append(Jaccard_SynonymyDetector.jaccard(probe_phrase, phrase2[0], 3))

        if nb_results == 1:
            # Нужна только 1 лучшая фраза
            best_index = np.argmax(y)
            best_phrase = phrases[best_index][0]
            best_sim = y[best_index]
            return best_phrase, best_sim
        else:
            # нужно вернуть nb_results ближайших фраз.
            sim = y
            phrases2 = [(phrases[i][0], sim[i]) for i in range(len(sim))]
            phrases2 = sorted(phrases2, key=lambda z: -z[1])
            return phrases2[:nb_results]

# coding: utf-8

from __future__ import print_function
import pickle
import os
import collections


def get_shingles(phrase):
    return set((z1 + z2 + z3) for z1, z2, z3 in zip(phrase, phrase[1:], phrase[2:]))


# https://www.programcreek.com/python/example/94974/Levenshtein.distance
def levenshtein_distance(a, b):
    """Return the Levenshtein edit distance between two strings *a* and *b*."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not a:
        return len(b)
    previous_row = range(len(b) + 1)
    for i, column1 in enumerate(a):
        current_row = [i + 1]
        for j, column2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (column1 != column2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# TODO: сделать базовый класс
class IntentDetector(object):
    def __init__(self):
        self.model = None
        self.nlp_transform = None

    def load(self, model_dir):
        model_path = os.path.join(model_dir, 'intent_classifier.model')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.nlp_transform = self.model['nlp_transform']

    def detect_intent(self, phrase_str, text_utils, word_embeddings):
        nphrase = phrase_str.lower()

        if nphrase in self.model['phrase2label']:
            # Фраза находится в lookup-таблице, обойдемся без классификации.
            return self.model['phrase2label'][nphrase]

        nphrase = text_utils.wordize_text(phrase_str)
        if nphrase in self.model['phrase2label']:
            return self.model['phrase2label'][nphrase]

        # Попробуем нечеткое сопоставление (коэф-т Жаккара), чтобы
        # учесть небольшие варианты написания ключевых фраз типа "неет"
        shingle2keyphrases = self.model['shingle2phrases']
        keyphrase2count = collections.Counter()
        shingles1 = get_shingles(u'['+nphrase+u']')
        for shingle in shingles1:
            if shingle in shingle2keyphrases:
                keyphrase2count.update(shingle2keyphrases[shingle])
        if len(keyphrase2count) > 0:
            for top_keyphrase, _ in keyphrase2count.most_common(5):
                #shingles2 = get_shingles(top_keyphrase)
                #jaccard = float(len(shingles1 & shingles2)) / float(len(shingles1 | shingles2))
                #if jaccard > 0.95:
                #    self.model['phrase2label'][nphrase]
                ldist = levenshtein_distance(top_keyphrase, nphrase)
                if ldist < 2:
                    return self.model['phrase2label'][top_keyphrase]

        if self.nlp_transform == 'lower':
            X_query = self.model['vectorizer'].transform([phrase_str.lower()])
        else:
            X_query = self.model['vectorizer'].transform([phrase_str])

        y_query = self.model['estimator'].predict(X_query)
        intent_index = y_query[0]
        intent_name = self.model['index2label'][intent_index]
        return intent_name



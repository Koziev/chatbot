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


class IntentDetector(object):
    """Классификаторы реплики: интент, сентимент и т.д."""
    def __init__(self):
        self.models = []

    def load(self, model_dir):
        for p in ['intent', 'abusive', 'sentiment', 'direction']:
            model_path = os.path.join(model_dir, '{}_classifier.model'.format(p))
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            nlp_transform = model['nlp_transform']
            self.models.append((model, nlp_transform))

    def append_intent(self, intents, intent):
        if not intent.startswith('0'):
            intents.append(intent)

    def detect_intent(self, phrase_str, text_utils):
        """Для фразы phrase_str вернет набор меток из классификаторов"""
        nphrase = phrase_str.lower()
        word_embeddings = text_utils.word_embeddings

        intents = []

        for model, nlp_transform in self.models:
            if nphrase in model['phrase2label']:
                # Фраза находится в lookup-таблице, обойдемся без классификации.
                self.append_intent(intents, model['phrase2label'][nphrase])
                continue

            nphrase = text_utils.wordize_text(phrase_str)
            if nphrase in model['phrase2label']:
                self.append_intent(intents, model['phrase2label'][nphrase])
                continue

            intent_detected = False

            # Попробуем нечеткое сопоставление (коэф-т Жаккара), чтобы
            # учесть небольшие варианты написания ключевых фраз типа "неет"
            shingle2keyphrases = model['shingle2phrases']
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
                        self.append_intent(intents, model['phrase2label'][top_keyphrase])
                        intent_detected = True
                        break

            if not intent_detected:
                if nlp_transform == 'lower':
                    X_query = model['vectorizer'].transform([phrase_str.lower()])
                else:
                    X_query = model['vectorizer'].transform([phrase_str])

                y_query = model['estimator'].predict(X_query)
                intent_index = y_query[0]
                intent_name = model['index2label'][intent_index]
                self.append_intent(intents, intent_name)

        return intents

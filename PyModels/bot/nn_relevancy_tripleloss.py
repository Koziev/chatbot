# -*- coding: utf-8 -*-

import json
import os
import logging
import itertools
import math
import operator
import numpy as np

from keras.models import model_from_json

from relevancy_detector import RelevancyDetector


def v_cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom > 0:
        return np.dot(a, b) / denom
    else:
        return 0


class NN_RelevancyTripleLoss(RelevancyDetector):
    """
    Модель для определения релевантности предпосылки и вопроса на базе
    нейросеточного классификатора с triple loss в обучении.
    """
    def __init__(self):
        super(NN_RelevancyTripleLoss, self).__init__()
        self.logger = logging.getLogger('NN_RelevancyTripleLoss')

    def load(self, models_folder):
        self.logger.info('Loading NN_RelevancyTripleLoss model files')

        with open(os.path.join(models_folder, 'nn_relevancy_tripleloss.config'), 'r') as f:
            self.model_config = json.load(f)

        arch_filepath = os.path.join(models_folder, 'nn_relevancy_tripleloss.arch')
        weights_path = os.path.join(models_folder, 'nn_relevancy_tripleloss.weights')
        with open(arch_filepath, 'r') as f:
            self.model = model_from_json(f.read())

        self.model.load_weights(weights_path)

        self.max_wordseq_len = self.model_config['max_wordseq_len']
        self.w2v_path = self.model_config['w2v_path']
        self.wordchar2vector_path = self.model_config['wordchar2vector_path']
        self.word_dims = self.model_config['word_dims']
        self.net_arch = self.model_config['net_arch']
        self.padding = self.model_config['padding']

        self.w2v_filename = os.path.basename(self.w2v_path)

    def normalize_qline(self, phrase):
        return phrase.replace(u'?', u' ').replace(u'!', u' ').strip()

    def get_most_relevant(self, probe_phrase, phrases, text_utils, word_embeddings, nb_results=1):
        all_phrases = list(itertools.chain(map(operator.itemgetter(0), phrases), [self.normalize_qline(probe_phrase)]))
        nb_phrases = len(all_phrases)

        pad_func = text_utils.lpad_wordseq if self.padding == 'left' else text_utils.rpad_wordseq

        X_data = np.zeros((nb_phrases, self.max_wordseq_len, self.word_dims), dtype=np.float32)
        for iphrase, phrase in enumerate(all_phrases):
            words = pad_func(text_utils.tokenize(self.normalize_qline(phrase)), self.max_wordseq_len)
            word_embeddings.vectorize_words(self.w2v_filename, words, X_data, iphrase)

        y_pred = self.model.predict(x=X_data, verbose=0)

        # Теперь для каждой фразы мы знаем вектор
        phrase2v = dict()
        for i in range(nb_phrases):
            phrase = all_phrases[i]
            v = y_pred[i]
            phrase2v[phrase] = v

        # Можем оценить близость введенной фразы к каждой предпосылке
        phrase_rels = []
        v1 = phrase2v[probe_phrase]
        for phrase2, _, _ in phrases:
            v2 = phrase2v[phrase2]
            #dist = np.sum(np.square(v1 - v2))
            #rel = math.exp(-dist)
            rel = v_cosine(v1, v2)
            phrase_rels.append((phrase2, rel))

        phrase_rels = sorted(phrase_rels, key=lambda z: -z[1])

        if nb_results == 1:
            # возвращаем единственную запись с максимальной релевантностью.
            best_premise = phrase_rels[0][0]
            best_rel = phrase_rels[0][1]
            return best_premise, best_rel
        else:
            # возвращаем заданное кол-во наиболее релевантных записей.
            n = min(nb_results, nb_results)
            best_premises = [phrase_rels[i][0] for i in range(min(n, len(phrases)))]
            best_rels = [phrase_rels[i][1] for i in range(n)]
            return best_premises, best_rels

    def calc_relevancy1(self, premise, question, text_utils, word_embeddings):
        _, rel = self.get_most_relevant(self, question, [premise], text_utils, word_embeddings, nb_results=1)
        return rel

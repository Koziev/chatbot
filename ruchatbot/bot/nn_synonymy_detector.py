# -*- coding: utf-8 -*-
"""
Нейросетевая реализация модели определения семантической близости двух предложений.
"""

import os
import json
import numpy as np
import logging
from keras.models import model_from_json
from synonymy_detector import SynonymyDetector


class NN_SynonymyDetector(SynonymyDetector):
    def __init__(self):
        super(NN_SynonymyDetector, self).__init__()
        self.logger = logging.getLogger('NN_SynonymyDetector')
        self.model = None
        self.model_config = None

    def load(self, models_folder):
        self.logger.info('Loading NN_SynonymyDetector model files')

        arch_filepath = os.path.join(models_folder, 'nn_synonymy.arch')
        weights_path = os.path.join(models_folder, 'nn_synonymy.weights')
        with open(arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(weights_path)
        self.model = m

        with open(os.path.join(models_folder, 'nn_synonymy.config'), 'r') as f:
            self.model_config = json.load(f)

        self.word_dims = self.model_config['word_dims']
        self.w2v_path = self.model_config['w2v_path']
        self.max_wordseq_len = int(self.model_config['max_wordseq_len'])
        self.padding = self.model_config['padding']
        self.net_arch = self.model_config['net_arch']
        assert(self.net_arch == 'lstm(cnn)')

        self.w2v_filename = os.path.basename(self.w2v_path)

    def get_most_similar(self, probe_phrase, phrases, text_utils, word_embeddings, nb_results=1):
        assert(nb_results > 0)
        assert(len(probe_phrase) != 0)

        nb_phrases = len(phrases)
        X1_probe = np.zeros((nb_phrases, self.max_wordseq_len, self.word_dims), dtype=np.float32)
        X2_probe = np.zeros((nb_phrases, self.max_wordseq_len, self.word_dims), dtype=np.float32)

        # Векторизуем фразы
        for iphrase, phrase in enumerate(phrases):
            if iphrase == 0:
                if self.padding == 'right':
                    words = text_utils.rpad_wordseq(text_utils.tokenize(probe_phrase), self.max_wordseq_len)
                else:
                    words = text_utils.lpad_wordseq(text_utils.tokenize(probe_phrase), self.max_wordseq_len)

                word_embeddings.vectorize_words(self.w2v_filename, words, X1_probe, iphrase)
            else:
                X1_probe[iphrase, :] = X1_probe[0, :]

            if self.padding == 'right':
                words = text_utils.rpad_wordseq(text_utils.tokenize(phrase[0]), self.max_wordseq_len)
            else:
                words = text_utils.lpad_wordseq(text_utils.tokenize(phrase[0]), self.max_wordseq_len)

            word_embeddings.vectorize_words(self.w2v_filename, words, X2_probe, iphrase)

        # Прогоняем через модель
        y = self.model.predict({'input_words1': X1_probe, 'input_words2': X2_probe})

        if nb_results == 1:
            # Нужна только 1 лучшая фраза
            best_index = np.argmax(y[:, 1])
            best_phrase = phrases[best_index][0]
            best_sim = y[best_index, 1]
            return best_phrase, best_sim
        else:
            # нужно вернуть nb_results ближайших фраз.
            sim = y[:, 1]
            phrases2 = [(phrases[i][0], sim[i]) for i in range(len(sim))]
            phrases2 = sorted(phrases2, key=lambda z: -z[1])
            return phrases2[:nb_results]

# -*- coding: utf-8 -*-
"""
Аппликатор модели wordchar2vector для чатбота.
Используется для генерации векторов тех слов, которые заранее
не обработаны (см. скрипт wordchar2vector.py в режиме --train 0 --vectorize 1)
"""

from __future__ import print_function

import numpy as np
import os
import json
import logging

# Для генерации символьных эмбеддингов новых слов нужна будет
# обученная модель wordchar2vec.
from keras.models import model_from_json


class Wordchar2VectorModel:
    def __init__(self):
        self.logger = logging.getLogger('Wordchar2VectorModel')
        self.model = None
        self.model_config = None
        self.word2vector = dict()

    def load(self, models_folder):
        self.logger.info('Loading Wordchar2VectorModel model files')

        with open(os.path.join(models_folder, 'wordchar2vector.config'), 'r') as f:
            self.model_config = json.load(f)

        # грузим готовую модель
        arch_filepath = os.path.join(models_folder, os.path.basename(self.model_config['arch_filepath']))
        with open(arch_filepath, 'r') as f:
            self.model = model_from_json(f.read())

        weights_path = os.path.join(models_folder, os.path.basename(self.model_config['weights_path']))
        self.model.load_weights(weights_path)

        # прочие параметры
        self.vec_size = self.model_config['vec_size']
        self.max_word_len = self.model_config['max_word_len']
        self.char2index = self.model_config['char2index']
        self.nb_chars = len(self.char2index)

    def vectorize_word(self, word, X_batch, irow):
        for ich, ch in enumerate(word):
            if ch not in self.char2index:
                self.logger.error(u'Char "{}" code={} word="{}" missing in char2index'.format(ch, ord(ch), word))
            else:
                X_batch[irow, ich] = self.char2index[ch]

    def build_vector(self, word):
        if word in self.word2vector:
            return self.word2vector[word]

        X_data = np.zeros((1, self.max_word_len + 2), dtype=np.int32)
        self.vectorize_word(word, X_data, 0)

        y_pred = self.model.predict(x=X_data, verbose=0)

        word_vect = y_pred[0, :]
        self.word2vector[word] = word_vect
        return word_vect

# -*- coding: utf-8 -*-
"""
Entity extraction engine для чатбота
"""

import os
import json
import numpy as np
import logging
import itertools
import random
import pickle

from keras.models import model_from_json
import keras_contrib
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from utils.padding_utils import PAD_WORD, lpad_wordseq, rpad_wordseq


class EntityExtractor(object):
    def __init__(self):
        self.logger = logging.getLogger('EntityExtractor')

    def load(self, models_folder):
        self.logger.info(u'Loading EntityExtractor model files from "{}"'.format(models_folder))

        config_path = os.path.join(models_folder, 'nn_entity_extractor.config')

        with open(config_path, 'r') as f:
            model_config = json.load(f)
            self.max_inputseq_len = model_config['max_inputseq_len']
            self.w2v_path = model_config['word2vector_path']
            #wordchar2vector_path = model_config['wordchar2vector_path']
            self.word_dims = model_config['word_dims']
            self.padding = model_config['padding']
            self.index2entity = model_config['index2entity']

        self.w2v_filename = os.path.basename(self.w2v_path)
        self.models = dict()
        self.entity2index = dict()

        for index, entity in self.index2entity.items():
            self.entity2index[entity] = int(index)
            arch_filepath = os.path.join(models_folder, 'nn_entity_extractor.({}).arch'.format(index))
            weights_path = os.path.join(models_folder, 'nn_entity_extractor.({}).weights'.format(index))

            with open(arch_filepath, 'r') as f:
                model = model_from_json(f.read(), {'CRF': CRF})

            model.load_weights(weights_path)
            self.models[int(index)] = model

        self.X_probe = np.zeros((1, self.max_inputseq_len, self.word_dims), dtype='float32')

        pass

    def extract_entity(self, entity_name, phrase, text_utils, embeddings):
        model = self.models[self.entity2index[entity_name]]

        self.X_probe.fill(0)

        words = text_utils.tokenize(phrase)
        if self.padding == 'right':
            words = rpad_wordseq(words, self.max_inputseq_len)
        else:
            words = lpad_wordseq(words, self.max_inputseq_len)

        embeddings.vectorize_words(self.w2v_filename, words, self.X_probe, 0)

        inputs = dict()
        inputs['input'] = self.X_probe

        y = model.predict(x=inputs)[0]
        predicted_labels = np.argmax(y, axis=-1)

        selected_words = [word for word, label in zip(words, predicted_labels) if label == 1]
        entity_text = u' '.join(selected_words).strip()
        return entity_text



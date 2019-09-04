# -*- coding: utf-8 -*-
"""
Модель для формирования списка команд для генеративной грамматики.
"""

from __future__ import division
from __future__ import print_function

import itertools
import json
import os
import numpy as np

from keras.models import model_from_json
import keras_contrib
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


PAD_WORD = u''
BEG_TOKEN = '<begin>'
END_TOKEN = '<end>'


def lpad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def pad_wordseq(words, n, padding):
    if padding == 'right':
        return rpad_wordseq(words, n)
    else:
        return lpad_wordseq(words, n)


class NN_AnswerGenerator2(object):
    def __init__(self):
        pass

    def load(self, model_folder):
        config_path = os.path.abspath(os.path.join(model_folder, 'nn_answer_generator_new2.config'))

        with open(config_path, 'r') as f:
            model_config = json.load(f)
            self.index2label = dict(model_config['index2label'])
            self.index2term = dict(model_config['index2term'])
            self.max_inputseq_len = model_config['max_inputseq_len']
            self.word_dims = model_config['word_dims']
            self.padding = model_config['padding']
            self.arch = model_config['arch']
            self.w2v_filename = os.path.basename(model_config['w2v_path'])
            arch_file = model_config['arch_file']
            weights_file = model_config['weights']

        assert(self.arch == 'crf')

        weights_path = os.path.abspath(os.path.join(model_folder, os.path.basename(weights_file)))
        arch_path = os.path.abspath(os.path.join(model_folder, os.path.basename(arch_file)))

        with open(arch_path, 'r') as f:
            self.model = model_from_json(f.read(), {'CRF': CRF})

        self.model.load_weights(weights_path)

        self.X1_probe = np.zeros((1, self.max_inputseq_len+2, self.word_dims), dtype='float32')
        self.X2_probe = np.zeros((1, self.max_inputseq_len+2, self.word_dims), dtype='float32')

    def predict(self, premises, question, word2vec):
        assert(len(premises) == 1)

        self.X1_probe.fill(0)
        self.X2_probe.fill(0)

        # Векторизуем входные данные - предпосылки и вопрос
        words = pad_wordseq(premises[0], self.max_inputseq_len+2, self.padding)
        word2vec.vectorize_words(self.w2v_filename, words, self.X1_probe, 0)

        words = pad_wordseq(question, self.max_inputseq_len+2, self.padding)
        word2vec.vectorize_words(self.w2v_filename, words, self.X2_probe, 0)

        inputs = dict()
        inputs['input1'] = self.X1_probe
        inputs['input2'] = self.X2_probe
        y_pred = self.model.predict(x=inputs, verbose=0)

        if self.arch == 'crf':
            terms = np.argmax(y_pred[0], axis=-1)
            terms = [self.index2term[i] for i in terms]
            terms = [t for t in terms if t not in (BEG_TOKEN, END_TOKEN, PAD_WORD)]

        template_str = u' '.join(terms).strip()
        return template_str

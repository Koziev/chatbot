# -*- coding: utf-8 -*-
"""
Модель для определения длины ответа
"""

from __future__ import division
from __future__ import print_function

import itertools
import json
import os
import numpy as np

from keras.models import model_from_json


PAD_WORD = u''
padding = 'left'


def pad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


class AnswerLengthPredictor(object):
    def __init__(self):
        pass

    def load(self, model_folder):
        arch_filepath = os.path.join(model_folder, 'nn_answer_length.arch')
        weights_path = os.path.join(model_folder, 'nn_answer_length.weights')
        config_path = os.path.join(model_folder, 'nn_answer_length.config')

        with open(config_path, 'r') as f:
            model_config = json.load(f)
            self.max_inputseq_len = model_config['max_inputseq_len']
            self.w2v_path = os.path.basename(model_config['word2vector_path'])
            #wordchar2vector_path = model_config['wordchar2vector_path']
            self.word_dims = model_config['word_dims']
            self.max_nb_premises = model_config['max_nb_premises']

        with open(arch_filepath, 'r') as f:
            self.model = model_from_json(f.read())

        self.model.load_weights(weights_path)

        self.Xn_probe = []
        for i in range(self.max_nb_premises + 1):
            self.Xn_probe.append(np.zeros((1, self.max_inputseq_len, self.word_dims), dtype='float32'))
        self.X_word = np.zeros((1, self.word_dims), dtype='float32')

    def predict(self, premises, question, word2vec):
        # Очистим входные тензоры перед заполнением новыми данными
        for X in self.Xn_probe:
            X.fill(0)

        # Векторизуем входные данные - предпосылки и вопрос
        for ipremise, words in enumerate(premises):
            words = pad_wordseq(words, self.max_inputseq_len)
            #vectorize_words(words, self.Xn_probe[ipremise], 0, word2vec)
            word2vec.vectorize_words(self.w2v_path, words, self.Xn_probe[ipremise], 0)

        words = pad_wordseq(question, self.max_inputseq_len)
        #vectorize_words(words, self.Xn_probe[self.max_nb_premises], 0, word2vec)
        word2vec.vectorize_words(self.w2v_path, words, self.Xn_probe[self.max_nb_premises], 0)

        inputs = dict()
        for ipremise in range(self.max_nb_premises):
            inputs['premise{}'.format(ipremise)] = self.Xn_probe[ipremise]
        inputs['question'] = self.Xn_probe[self.max_nb_premises]
        inputs['word'] = self.X_word

        y_probe = self.model.predict(x=inputs)[0]
        return dict((i+1, y_probe[i]) for i in range(y_probe.shape[0]))

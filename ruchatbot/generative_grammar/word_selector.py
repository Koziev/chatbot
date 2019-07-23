# -*- coding: utf-8 -*-
"""
Word attention model
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



class WordSelector(object):
    def __init__(self):
        pass

    def load(self, model_folder):
        config_path = os.path.join(model_folder, 'nn_word_selector.config')
        arch_filepath = os.path.join(model_folder, 'nn_word_selector.arch')
        weights_path = os.path.join(model_folder, 'nn_word_selector.weights')

        self.stop_words = set(u'кстати пожалуй кто что куда откуда почему зачем как сколько когда насколько где же ли а ну бы б ж . ! ? - да нет'.split())

        with open(config_path, 'r') as f:
            model_config = json.load(f)
            self.max_inputseq_len = model_config['max_inputseq_len']
            self.w2v_path = os.path.basename(model_config['w2v_path'])
            #wordchar2vector_path = model_config['wordchar2vector_path']
            self.word_dims = model_config['word_dims']
            self.max_nb_premises = model_config['max_nb_premises']

        with open(arch_filepath, 'r') as f:
            self.model = model_from_json(f.read())

        self.model.load_weights(weights_path)

        # приготовим матрицы для обработки одного сэмпла
        self.Xn_probe = []
        for i in range(self.max_nb_premises + 1):
            self.Xn_probe.append(np.zeros((1, self.max_inputseq_len, self.word_dims), dtype='float32'))
        self.X_word = np.zeros((1, self.word_dims), dtype='float32')

    def get_w2v_path(self):
        return self.w2v_path

    def select_words(self, premises, question, word2vec):
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

        # Идем по всем словам в предпосылках и в вопросе, получаем вероятность
        # их участия в генерации.
        input_words = set()
        for words in itertools.chain(premises, [question]):
            input_words.update(words)

        word_p = []
        for probe_word in input_words:
            self.X_word.fill(0)

            if probe_word in self.stop_words:
                continue

            # Вход для проверяемого слова:
            #if probe_word not in word2vec:
            #    logging.error(u'word "{}" is missing in word2vec'.format(probe_word))
            #else:
            self.X_word[0, :] = word2vec.vectorize_word1(self.w2v_path, probe_word)

            inputs = dict()
            for ipremise in range(self.max_nb_premises):
                inputs['premise{}'.format(ipremise)] = self.Xn_probe[ipremise]
            inputs['question'] = self.Xn_probe[self.max_nb_premises]
            inputs['word'] = self.X_word

            y_probe = self.model.predict(x=inputs)
            p = y_probe[0][1]
            word_p.append((probe_word, p))

        return word_p

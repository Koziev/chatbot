# -*- coding: utf-8 -*-
"""
Модель для определения релевантности ответа при заданных предпосылках и вопросе
"""

from __future__ import division
from __future__ import print_function

import itertools
import json
import os
import io
import numpy as np
import logging

from keras.models import model_from_json


PAD_WORD = u''
padding = 'left'


def pad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


class AnswerRelevancy(object):
    def __init__(self):
        pass

    def load(self, model_folder):
        config_path = os.path.join(model_folder, 'nn_answer_relevancy.config')
        arch_filepath = os.path.join(model_folder, 'nn_answer_relevancy.arch')
        weights_path = os.path.join(model_folder, 'nn_answer_relevancy.weights')

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

    def score_answers(self, premises, question, answers, word2vec, tokenizer, len2proba):
        scored_answers = []

        self.nb_samples = len(answers)

        Xn_probe = []
        for i in range(self.max_nb_premises + 1):
            Xn_probe.append(np.zeros((self.nb_samples, self.max_inputseq_len, self.word_dims), dtype='float32'))

        X_answer = np.zeros((self.nb_samples, self.max_inputseq_len, self.word_dims), dtype='float32')

        # Векторизуем входные данные - предпосылки, вопрос, варианты ответа.
        # Для всех ответов у нас один набор предпосылок и вопрос.
        for ianswer, answer in enumerate(answers):
            if True:  #ianswer == 0:
                for ipremise, words in enumerate(premises):
                    words = pad_wordseq(words, self.max_inputseq_len)
                    #vectorize_words(words, Xn_probe[ipremise], ianswer, word2vec)
                    word2vec.vectorize_words(self.w2v_path, words, Xn_probe[ipremise], ianswer)

                words = pad_wordseq(question, self.max_inputseq_len)
                #vectorize_words(words, Xn_probe[self.max_nb_premises], ianswer, word2vec)
                word2vec.vectorize_words(self.w2v_path, words, Xn_probe[self.max_nb_premises], ianswer)
            else:
                # копируем из первого сэмпла
                for i in range(self.max_nb_premises):
                    Xn_probe[i][ianswer, :, :] = Xn_probe[i][0, :, :]

            words = pad_wordseq([token.word for token in answer.get_words()], self.max_inputseq_len)
            #vectorize_words(words, X_answer, ianswer, word2vec)
            word2vec.vectorize_words(self.w2v_path, words, X_answer, ianswer)

        # Прогоняем подготовленные тензоры через модель, для каждого варианта ответа
        # получим его вероятность (точнее вес).
        inputs = dict()
        for ipremise in range(self.max_nb_premises):
            inputs['premise{}'.format(ipremise)] = Xn_probe[ipremise]
        inputs['question'] = Xn_probe[self.max_nb_premises]
        inputs['answer'] = X_answer

        y_probe = self.model.predict(x=inputs)

        for ianswer, answer in enumerate(answers):
            #if len(answer.words) == 1 and answer.words[0] in (u'конец', u'кембридж'):
            #    print('DEBUG@270')
            p_total = answer.get_rank() * len2proba.get(len(answers[ianswer].words), 0) * y_probe[ianswer, 1]
            answer.set_rank(p_total)
            scored_answers.append(answer)

        return scored_answers

    def score_answer(self, premises, question, answer, word2vec):
        self.nb_samples = 1
        Xn_probe = []
        for i in range(self.max_nb_premises + 1):
            Xn_probe.append(np.zeros((self.nb_samples, self.max_inputseq_len, self.word_dims), dtype='float32'))
        X_answer = np.zeros((self.nb_samples, self.max_inputseq_len, self.word_dims), dtype='float32')

        # Очистим входные тензоры перед заполнением новыми данными
        X_answer.fill(0)
        for X in Xn_probe:
            X.fill(0)

        # Векторизуем входные данные - предпосылки и вопрос
        for ipremise, words in enumerate(premises):
            words = pad_wordseq(words, self.max_inputseq_len)
            word2vec.vectorize_words(self.w2v_path, words, Xn_probe[ipremise], 0)

        words = pad_wordseq(question, self.max_inputseq_len)
        word2vec.vectorize_words(self.w2v_path, words, Xn_probe[self.max_nb_premises], 0)

        words = pad_wordseq(answer, self.max_inputseq_len)
        word2vec.vectorize_words(self.w2v_path, words, X_answer, 0)

        inputs = dict()
        for ipremise in range(self.max_nb_premises):
            inputs['premise{}'.format(ipremise)] = Xn_probe[ipremise]
        inputs['question'] = Xn_probe[self.max_nb_premises]
        inputs['answer'] = X_answer

        y_probe = self.model.predict(x=inputs)
        p = y_probe[0][1]
        return p

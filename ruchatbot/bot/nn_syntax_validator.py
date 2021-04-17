# -*- coding: utf-8 -*-
"""
04-01-2021 первая версия модели с использованием BPE токенизации
17-04-2021 добавлена проверка на повтор слов, чтобы детектировать типовую проблему генеративной seq2seq модели нтерпретатора
"""


import os
import json
import logging
import collections

import numpy as np
from keras.models import model_from_json
from keras import backend as K
import sentencepiece as spm
import tensorflow as tf  # 13-05-2019

from ruchatbot.bot.model_applicator import ModelApplicator


class NN_SyntaxValidator(ModelApplicator):
    def __init__(self):
        super(NN_SyntaxValidator, self).__init__()
        self.logger = logging.getLogger('NN_SyntaxValidator')

    def load(self, models_folder):
        self.logger.info('Loading NN_SyntaxValidator model files')

        with open(os.path.join(models_folder, 'nn_syntax_validator.config'), 'r') as f:
            model_config = json.load(f)

        self.max_inputseq_len = model_config['max_wordseq_len']
        self.token2index = model_config['token2index']
        self.arch_filepath = self.get_model_filepath(models_folder, model_config['arch_filepath'])
        self.weights_filepath = self.get_model_filepath(models_folder, model_config['weights_path'])

        self.bpe_model = spm.SentencePieceProcessor()
        rc = self.bpe_model.Load(self.get_model_filepath(models_folder, model_config['bpe_model_name']+'.model'))
        self.logger.debug('NN_SyntaxValidator::bpe_model loaded with status=%d', rc)

        #self.graph = tf.Graph()
        #self.tf_sess = tf.Session(graph=self.graph)
        #self.tf_sess.__enter__()

        with open(self.arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(self.weights_filepath)
        self.model = m

        self.graph = tf.compat.v1.get_default_graph() # эксперимент с багом 13-05-2019

        # начало отладки
        #self.model.summary()
        # конец отладки

        self.X_probe = np.zeros((1, self.max_inputseq_len,), dtype=np.int32)
        self.inputs = [self.X_probe]

    def is_valid(self, text0, text_utils):
        # препроцессинг текста должен быть идентичен тому, который используется при обучении модели
        words = text_utils.tokenize(text0)
        if len(words) < 1:
            return 0.0

        word2count = collections.Counter(words)
        if word2count.most_common(1)[1] >=4:
            # Есть слово, которое повторено 4 и более раз
            return 0.0

        text = text_utils.remove_terminators(' '.join(words))

        # Очищаем содержимое входных тензоров от результатов предыдущего расчета
        self.X_probe.fill(0)

        tx = self.bpe_model.EncodeAsPieces(text)
        for itoken, token in enumerate(tx[:self.max_inputseq_len]):
            self.X_probe[0, itoken] = self.token2index.get(token, 0)

        #with self.graph.as_default():
        #    y = self.model.predict(x=self.inputs)[0]
        y = self.model.predict(x=self.inputs)[0]

        p_valid = y[0]

        if p_valid < 0.5:
            self.logger.debug('NN_SyntaxValidator text="%s" p_valid=%f', text, p_valid)

        return p_valid

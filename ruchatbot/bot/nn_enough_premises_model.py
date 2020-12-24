# -*- coding: utf-8 -*-
"""
24-12-2020 переработка модели: токенизация BPE, обучаемые эмбеддинги токенов.
"""


import os
import json
import logging

import numpy as np
from keras.models import model_from_json
from keras import backend as K
import sentencepiece as spm
import tensorflow as tf  # 13-05-2019

from ruchatbot.bot.enough_premises_model import EnoughPremisesModel


class NN_EnoughPremisesModel(EnoughPremisesModel):
    """
    Нейросетевая реализация модели определения достаточности набора
    предпосылок для ответа на вопрос. 
    """

    def __init__(self):
        super(NN_EnoughPremisesModel, self).__init__()
        self.logger = logging.getLogger('NN_EnoughPremisesModel')

    def load(self, models_folder):
        self.logger.info('Loading NN_EnoughPremisesModel model files')

        with open(os.path.join(models_folder, 'nn_enough_premises.config'), 'r') as f:
            model_config = json.load(f)

        self.max_inputseq_len = model_config['max_inputseq_len']
        self.max_nb_premises = model_config['max_nb_premises']
        self.token2index = model_config['token2index']
        self.arch_filepath = self.get_model_filepath(models_folder, model_config['arch_filepath'])
        self.weights_filepath = self.get_model_filepath(models_folder, model_config['weights_path'])

        self.bpe_model = spm.SentencePieceProcessor()
        rc = self.bpe_model.Load(self.get_model_filepath(models_folder, model_config['bpe_model_name']+'.model'))
        self.logger.debug('NN_EnoughPremisesModel.bpe_model loaded with status=%d', rc)

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

        self.Xn_probe = []
        for _ in range(self.max_nb_premises+1):
            x = np.zeros((1, self.max_inputseq_len,), dtype=np.int32)
            self.Xn_probe.append(x)

        self.inputs = dict()
        for ipremise in range(self.max_nb_premises):
            self.inputs['premise{}'.format(ipremise)] = self.Xn_probe[ipremise]
        self.inputs['question'] = self.Xn_probe[self.max_nb_premises]

    def is_enough(self, premise_str_list, question_str, text_utils):
        assert(len(premise_str_list) <= self.max_nb_premises)
        assert(len(question_str) > 0)

        # Очищаем содержимое входных тензоров от результатов предыдущего расчета
        for i in range(self.max_nb_premises+1):
            self.Xn_probe[i].fill(0)

        # Заполняем входные тензоры векторами слов предпосылок и вопроса.
        for ipremise, premise in enumerate(premise_str_list):
            tx = self.bpe_model.EncodeAsPieces(premise)
            for itoken, token in enumerate(tx):
                self.Xn_probe[ipremise][0, itoken] = self.token2index.get(token, 0)

        tx = self.bpe_model.EncodeAsPieces(question_str)
        for itoken, token in enumerate(tx):
            self.Xn_probe[self.max_nb_premises][0, itoken] = self.token2index.get(token, 0)

        #with self.graph.as_default():
        #    y = self.model.predict(x=self.inputs)[0]
        y = self.model.predict(x=self.inputs)[0]

        p_enough = y[0]
        return p_enough

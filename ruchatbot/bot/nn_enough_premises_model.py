# -*- coding: utf-8 -*-

import os
import json
import logging
import numpy as np
from keras.models import model_from_json
from keras import backend as K

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
        self.w2v_path = model_config['w2v_path']
        self.wordchar2vector_path = model_config['wordchar2vector_path']
        self.padding = model_config['padding']
        self.word_dims = model_config['word_dims']
        self.max_nb_premises = model_config['max_nb_premises']
        self.arch_filepath = self.get_model_filepath(models_folder, model_config['arch_filepath'])
        self.weights_filepath = self.get_model_filepath(models_folder, model_config['weights_path'])

        with open(self.arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(self.weights_filepath)
        self.model = m

        self.graph = tf.get_default_graph() # эксперимент с багом 13-05-2019

        # начало отладки
        #self.model.summary()
        # конец отладки


        self.w2v_filename = os.path.basename(self.w2v_path)

        self.Xn_probe = []
        for _ in range(self.max_nb_premises+1):
            x = np.zeros((1, self.max_inputseq_len, self.word_dims), dtype=np.float32)
            self.Xn_probe.append(x)

        self.inputs = dict()
        for ipremise in range(self.max_nb_premises):
            self.inputs['premise{}'.format(ipremise)] = self.Xn_probe[ipremise]
        self.inputs['question'] = self.Xn_probe[self.max_nb_premises]

    def is_enough(self, premise_str_list, question_str, text_utils, word_embeddings):
        assert(len(premise_str_list) <= self.max_nb_premises)
        assert(len(question_str) > 0)

        # Очищаем содержимое входных тензоров от результатов предыдущего расчета
        for i in range(self.max_nb_premises+1):
            self.Xn_probe[i].fill(0)

        # Заполняем входные тензоры векторами слов предпосылок и вопроса.
        for ipremise, premise in enumerate(premise_str_list):
            if self.padding == 'right':
                words = text_utils.rpad_wordseq(text_utils.tokenize(premise), self.max_inputseq_len)
            else:
                words = text_utils.lpad_wordseq(text_utils.tokenize(premise), self.max_inputseq_len)
            word_embeddings.vectorize_words(self.w2v_filename, words, self.Xn_probe[ipremise], 0)

        if self.padding == 'right':
            words = text_utils.rpad_wordseq(text_utils.tokenize(question_str), self.max_inputseq_len)
        else:
            words = text_utils.lpad_wordseq(text_utils.tokenize(question_str), self.max_inputseq_len)
        word_embeddings.vectorize_words(self.w2v_filename, words, self.Xn_probe[self.max_nb_premises], 0)

        with self.graph.as_default():
            y = self.model.predict(x=self.inputs)[0]

        p_enough = y[0]
        return p_enough

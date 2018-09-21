# -*- coding: utf-8 -*-
"""
Реализация классификатора режима генерации ответа в чатботе на базе нейросетки.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

import os
import json
import numpy as np
import logging
from keras.models import model_from_json
from model_selector import ModelSelector

class NN_ModelSelector(ModelSelector):
    def __init__(self):
        super(NN_ModelSelector, self).__init__()
        self.logger = logging.getLogger('NN_ModelSelector')

    def load(self, models_folder):
        self.logger.info('Loading NN_ModelSelector model files')

        arch_filepath = os.path.join(models_folder, 'qa_model_selector.arch')
        weights_path = os.path.join(models_folder, 'qa_model_selector.weights')
        with open(arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(weights_path)
        self.model = m

        with open(os.path.join(models_folder, 'qa_model_selector.config'), 'r') as f:
            self.model_config = json.load(f)

        self.word_dims = self.model_config['word_dims']
        self.w2v_path = self.model_config['w2v_path']
        self.padding = self.model_config['padding']
        self.max_inputseq_len = self.model_config['max_inputseq_len']
        self.max_nb_premises = self.model_config['max_nb_premises']

        self.w2v_filename = os.path.basename(self.w2v_path)
        self.Xn_probe = []
        for _ in range(self.max_nb_premises+1):
            x = np.zeros((1, self.max_inputseq_len, self.word_dims), dtype=np.float32)
            self.Xn_probe.append(x)

        self.inputs = dict()
        for ipremise in range(self.max_nb_premises):
            self.inputs['premise{}'.format(ipremise)] = self.Xn_probe[ipremise]
        self.inputs['question'] = self.Xn_probe[self.max_nb_premises]


    def select_model(self, premise_str_list, question_str, text_utils, word_embeddings):
        """Определяем способ генерации ответа"""

        assert(len(premise_str_list) <= self.max_nb_premises)
        assert(len(question_str) > 0)

        # Очищаем содержимое входных тензоров от результатов предыдущего расчета
        for i in range(self.max_nb_premises+1):
            self.Xn_probe[i].fill(0)

        # Заполняем входные тензоры векторами слов прелпосылок и вопроса.
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

        y_probe = self.model.predict(x=self.inputs)

        model_selector = np.argmax(y_probe[0])
        return model_selector

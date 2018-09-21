# -*- coding: utf-8 -*-
"""
Нейросетевая реализация модели изменения слов в предложении для
изменения грамматического лица основного предиката.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

import os
import json
import numpy as np
import logging
import pickle
from keras.models import model_from_json
from person_change_model import PersonChangeModel

class NN_PersonChange(PersonChangeModel):
    def __init__(self):
        super(NN_PersonChange, self).__init__()
        self.logger = logging.getLogger('NN_PersonChange')

    def load(self, models_folder):
        self.logger.info('Loading NN_PersonChange model files')

        # Упрощенная модель для работы с грамматическим лицом
        with open(os.path.join(models_folder, 'person_change_dictionary.pickle'), 'r') as f:
            model = pickle.load(f)

        self.w1s = model['word_1s']
        self.w2s = model['word_2s']
        self.person_change_1s_2s = model['person_change_1s_2s']
        self.person_change_2s_1s = model['person_change_2s_1s']

        # нейросетевой детектор изменяемых слов
        with open(os.path.join(models_folder, 'nn_changeable_words.config'), 'r') as f:
            self.person_change_model_config = json.load(f)

        arch_filepath = self.get_model_filepath(models_folder, self.person_change_model_config['arch_filepath'])
        weights_path = self.get_model_filepath(models_folder, self.person_change_model_config['weights_path'])
        with open(arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(weights_path)
        self.person_change_model = m

        self.word_dims = self.person_change_model_config['word_dims']
        self.max_wordseq_len = int(self.person_change_model_config['max_inputseq_len'])

        # todo создание входных тензоров для модели changeable_words
        pass



    def change_person(self, sentence_str, target_person, text_utils, word_embeddings):
        # текущая реализация - упрощенная, использует два списка замен для слов.
        inwords = text_utils.tokenize(sentence_str)
        outwords = []
        for word in inwords:
            if target_person == '2s' and word in self.w1s and word in self.person_change_1s_2s:
                outwords.append(self.person_change_1s_2s[word])
            elif target_person == '1s' and word in self.w2s and word in self.person_change_2s_1s:
                outwords.append(self.person_change_2s_1s[word])
            else:
                outwords.append(word)

        return u' '.join(outwords)


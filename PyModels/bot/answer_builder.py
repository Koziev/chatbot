# -*- coding: utf-8 -*-
"""
Группа моделей, выполняющих генерацию текста ответа при заданных текстах предпосылки
и вопроса.

Для проекта чат-бота https://github.com/Koziev/chatbot
"""

import json
import os
import logging
import numpy as np

from word_embeddings import WordEmbeddings
#from xgb_yes_no_model import XGB_YesNoModel
from nn_yes_no_model import NN_YesNoModel
from nn_model_selector import NN_ModelSelector
from nn_wordcopy3 import NN_WordCopy3
from xgb_answer_generator_model import XGB_AnswerGeneratorModel


class AnswerBuilder(object):
    def __init__(self):
        self.logger = logging.getLogger('AnswerBuilder')
        self.trace_enabled = True

    def load_models(self, models_folder):
        self.models_folder = models_folder

        # Модель для выбора ответов yes|no на базе XGB
        #self.yes_no_model = XGB_YesNoModel()
        #self.yes_no_model.load(models_folder)

        self.yes_no_model = NN_YesNoModel()
        self.yes_no_model.load(models_folder)

        # Модель для выбора способа генерации ответа
        self.model_selector = NN_ModelSelector()
        self.model_selector.load(models_folder)

        # нейросетевые модели для генерации ответа.
        self.word_copy_model = NN_WordCopy3()
        self.word_copy_model.load(models_folder)

        self.answer_generator = XGB_AnswerGeneratorModel()
        self.answer_generator.load(models_folder)

    def get_w2v_paths(self):
        paths = []

        if self.word_copy_model.w2v_path is not None:
            paths.append(self.word_copy_model.w2v_path)

        if self.model_selector.w2v_path is not None:
            paths.append(self.model_selector.w2v_path)

        if self.yes_no_model.w2v_path is not None:
            paths.append(self.yes_no_model.w2v_path)

        if self.answer_generator.w2v_path is not None:
            paths.append(self.answer_generator.w2v_path)

        return paths

    def build_answer_text(self, premises, question, text_utils, word_embeddings):
        # Определяем способ генерации ответа
        assert(len(premises) <= 1)
        premise = premises[0] if len(premises) == 1 else u''

        model_selector = self.model_selector.select_model(premise_str_list=premises,
                                                          question_str=question,
                                                          text_utils=text_utils,
                                                          word_embeddings=word_embeddings)
        if self.trace_enabled:
            self.logger.debug('model_selector={}'.format(model_selector))

        # Теперь применяем соответствующую модель к предпосылкам и вопросу.
        answer = u''

        if model_selector == 0:
            # Ответ генерируется через классификацию на 2 варианта yes|no
            y = self.yes_no_model.calc_yes_no(premises, question, text_utils, word_embeddings)
            if y < 0.5:
                answer = text_utils.language_resources[u'нет']
            else:
                answer = text_utils.language_resources[u'да']
        elif model_selector == 1:
            # ответ генерируется через копирование слов из предпосылки.
            answer = self.word_copy_model.generate_answer(premise,
                                                          question,
                                                          text_utils,
                                                          word_embeddings)
        else: # варианты 2 и 3
            # Ответ генерируется посимвольно.
            # Вариант 3 - особый случай, когда выдается строка из одних цифр
            answer = self.answer_generator.generate_answer(premise,
                                                           question,
                                                           text_utils,
                                                           word_embeddings)

        return answer

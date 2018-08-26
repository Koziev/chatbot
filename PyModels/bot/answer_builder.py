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
from xgb_yes_no_model import XGB_YesNoModel
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
        self.yes_no_model = XGB_YesNoModel()
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
        paths.append(self.word_copy_model.w2v_path)
        paths.append(self.model_selector.w2v_path)
        return paths

    def build_answer_text(self, premise, question, text_utils, word_embeddings):
        # Определяем способ генерации ответа
        model_selector = self.model_selector.select_model(premise_str=premise,
                                                          question_str=question,
                                                          text_utils=text_utils,
                                                          word_embeddings=word_embeddings)
        if self.trace_enabled:
            self.logger.debug('model_selector={}'.format(model_selector))

        answer = u''

        if model_selector == 0:
            # Ответ генерируется через классификацию на 2 варианта yes|no
            y = self.yes_no_model.calc_yes_no(premise, question, text_utils, word_embeddings)
            if y < 0.5:
                answer = u'нет'  # TODO: вынести во внешние ресурсы
            else:
                answer = u'да'  # TODO: вынести во внешние ресурсы
        elif model_selector == 1:
            # ответ генерируется через копирование слов из предпосылки.
            answer = self.word_copy_model.generate_answer(premise, question, text_utils, word_embeddings)
        else:
            # Ответ генерируется посимвольно.
            answer = self.answer_generator.generate_answer(premise, question, text_utils, word_embeddings)

        return answer

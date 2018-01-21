# -*- coding: utf-8 -*-
"""
Аппликатор для модели классификатора, который определяет способ генерации
ответа чат-бота.
"""

import os
from model_applicator import ModelApplicator

class ModelSelector(ModelApplicator):
    def __init__(self):
        pass

    def select_model(self, premise_str, question_str, text_utils, word_embeddings):
        """
        Данный метод должен быть перегружен в производном классе, реализуя конкретный
        алгоритм классификации способа генерации ответа на пару предпосылка (premise_str)
        + вопрос (question_str)
        :param premise_str: unicode строка с предпосылкой
        :param question_str: unicode строка с вопросом
        :param text_utils: экземпляр класса TextUtils
        :param word_embeddings: экземпляр класса WordEmbeddings
        :return: целое число 0...2, определяющее способ генерации ответа
        """
        raise NotImplemented()

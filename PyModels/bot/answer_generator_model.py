# -*- coding: utf-8 -*-
"""
Базовый класс для модели генерации ответа на основе предпосылки и вопроса.
"""

import os
from model_applicator import ModelApplicator

class AnswerGeneratorModel(ModelApplicator):
    def __init__(self):
        pass

    def generate_answer(self, premise_str, question_str, text_utils, word_embeddings):
        raise NotImplemented()

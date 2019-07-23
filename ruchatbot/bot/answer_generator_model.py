# -*- coding: utf-8 -*-
"""
Базовый класс для модели генерации ответа на основе предпосылки и вопроса.
"""

from ruchatbot.bot.model_applicator import ModelApplicator


class AnswerGeneratorModel(ModelApplicator):
    def __init__(self):
        super(AnswerGeneratorModel, self).__init__()
        pass

    def generate_answer(self, premise_str, question_str, text_utils, word_embeddings):
        raise NotImplementedError()

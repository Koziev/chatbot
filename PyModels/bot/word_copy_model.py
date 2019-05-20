# -*- coding: utf-8 -*-

from abc import abstractmethod

from bot.model_applicator import ModelApplicator


class WordCopyModel(ModelApplicator):
    def __init__(self):
        pass

    @abstractmethod
    def generate_answer(self, premise_str, question_str, text_utils, word_embeddings):
        raise NotImplementedError()

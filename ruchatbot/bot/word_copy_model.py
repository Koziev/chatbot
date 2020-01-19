# -*- coding: utf-8 -*-

from abc import abstractmethod

from ruchatbot.bot.model_applicator import ModelApplicator


class WordCopyModel(ModelApplicator):
    def __init__(self):
        pass

    @abstractmethod
    def generate_answer(self, premise_str, question_str, text_utils):
        raise NotImplementedError()

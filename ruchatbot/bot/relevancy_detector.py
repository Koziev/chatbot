# -*- coding: utf-8 -*-

from abc import abstractmethod

from ruchatbot.bot.model_applicator import ModelApplicator


class RelevancyDetector(ModelApplicator):
    """
    Класс предназначен для скрытия деталей вычисления релевантности
    предпосылок и вопроса.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_most_relevant(self, probe_phrase, phrases, text_utils, nb_results=1):
        raise NotImplemented()

    @abstractmethod
    def calc_relevancy1(self, premise, question, text_utils):
        raise NotImplemented()

    def get_w2v_path(self):
        return None

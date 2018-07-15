# -*- coding: utf-8 -*-

import os

from model_applicator import ModelApplicator

class RelevancyDetector(ModelApplicator):
    """
    Класс предназначен для скрытия деталей вычисления релевантности
    предпосылок и вопроса.
    """
    def __init__(self):
        pass

    def get_most_relevant(self, probe_phrase, phrases, text_utils, word_embeddings, nb_results=1):
        raise NotImplemented()

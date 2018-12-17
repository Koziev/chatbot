# -*- coding: utf-8 -*-

from model_applicator import ModelApplicator


class SynonymyDetector(ModelApplicator):
    """
    Интерфейс для детекторов синонимичности - моделей, которые
    определяют семантическую эквивалентность двух фраз. В отличие
    от семантической релевантности вопроса и предпосылки, реализуемой
    классом RelevancyDetector и его потомками, синонимичность
    требует примерно равного семантического объема двух фраз, то есть
    вопрос к части предпосылки не считается синонимом.
    """
    def __init__(self):
        super(SynonymyDetector, self).__init__()

    def get_most_similar(self, probe_phrase, phrases, text_utils, word_embeddings, nb_results=1):
        raise NotImplementedError()

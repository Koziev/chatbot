# -*- coding: utf-8 -*-

from ruchatbot.bot.model_applicator import ModelApplicator


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

    def calc_synonymy2(self, phrase1, phrase2, text_utils, word_embeddings):
        raise NotImplementedError()

    def get_threshold(self):
        """
        Возвращаемая моделью оценка синонимичности часто нужна не только для выбора лучшего
        варианта среди нескольких альтернатив. Обычно нужно еще понять, достаточно ли похожи
        фразы, чтобы считать их синонимичными. Возвращаемое этой функцией значение как
        раз используется по умолчанию как порог. В будущем можно инициализировать его
        из конфига при загрузке модели.
        """
        return 0.7

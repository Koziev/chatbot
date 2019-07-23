# -*- coding: utf-8 -*-

from ruchatbot.bot.model_applicator import ModelApplicator


class EnoughPremisesModel(ModelApplicator):
    """
    Модель выполняет оценку достаточности набора предпосылок для
    ответа на вопрос. Например, для арифметических вопросов типа
    'Сколько будет 2 плюс 1?' предпосылки не нужны, следовательно
    и поиск фактов в базе знаний выполнять не нужно.
    """
    def __init__(self):
        pass

    def is_enough(self, premise_str_list, question_str, text_utils, word_embeddings):
        """
        Определяем, достаточен ли набор предпосылок для ответа на вопрос
        :param premise_str_list: список unicode строк с предпосылками, может быть 0 строк.
        :param question_str: unicode строка с вопросом.
        :param text_utils: интерфейс для языковых процедур
        :param word_embeddings: интерфейс для доступа к векторам слов
        :return:
        """
        raise NotImplementedError()

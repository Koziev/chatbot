# -*- coding: utf-8 -*-

import os
from model_applicator import ModelApplicator

class WordCopyModel(ModelApplicator):
    """
    Базовый класс для моделей генерации ответа на вопрос через
    копирование слов из предпосылки.
    """
    def __init__(self):
        pass

    def copy_words(self, premise_words, question_words, text_utils, word_embeddings):
        """
        :param premise_words: список юникодных строк слов предпосылки
        :param question_words: список юникодных строк слов вопроса
        :param text_utils: экземпляр класс TextUtils
        :param word_embeddings: экземпляр класса WordEmbeddings
        :return: unicode строка с текстом ответа
        """
        raise NotImplemented()

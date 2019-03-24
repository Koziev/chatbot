# -*- coding: utf-8 -*-


class BaseFaqStorage(object):
    """Интерфейс доступа к FAQ базе"""

    def __init__(self):
        pass

    def get_most_similar(self, question_str, similarity_detector, word_embeddings, text_utils):
        """
        Ищем наиболее близкий вопрос в FAQ списке и возвращаем привязанный к
        этому вопросу текст (ответ).

        Вернем кортеж из элементов (best_answer, best_rel, best_question)
        """
        raise NotImplementedError()


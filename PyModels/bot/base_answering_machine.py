# -*- coding: utf-8 -*-

class BaseAnsweringMachine(object):
    """
    Базовый класс с интерфейсом для диалогового движка.
    """

    def __init__(self):
        pass


    def get_session_factory(self):
        """
        Производный класс должен перегрузить этот метод, чтобы
        он возвращал экземпляр класса, производного от BaseDialogSessionFactory
        :return:
        """
        raise NotImplemented()

    def start_conversation(self):
        pass

    def push_phrase(self, bot, interlocutor, phrase):
        """
        Обработка реплики или длительного молчания собеседника.

        :param bot: экземпляр бота
        :param interlocutor: строковый идентификатор собеседника,
                             например - ник в мессенджере.

        :param phrase: юникодный текст реплики от собеседника.
        """
        pass

    def pop_phrase(self, bot, interlocutor):
        """
        Получение очередной ответной реплики бота из буфера ответов.
        Так как ответных реплик в общем случае может быть несколько,
        то возвращается самый старый элемент буфера либо пустая строка
        при пустом буфере.

        :param bot: экземпляр бота
        :param interlocutor: строковый идентификатор собеседника,
                             например - ник в мессенджере.

        :return: текст ответной реплики или пустая строка, если буфер
                 ответов пуст.
        """
        return u''


    def get_session(self, interlocutor):
        return self.get_session_factory()[interlocutor]

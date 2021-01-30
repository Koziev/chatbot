# -*- coding: utf-8 -*-


class BaseDialogSessionFactory(object):
    """
    Базовый класс для управления диалоговыми сессиями - загрузка из
    хранилища, вопросы персистентности. В движке используется один из
    классов-наследников.
    """
    def __init__(self):
        pass

    def __getitem__(self, bot_id, interlocutor_id):
        """
        Получение объекта сессии по идентификатору собеседника.
        При необходимости сессия может быть создана и загружена из хранилища.
        :param bot_id: уникальный строковый идентификатор бота
        :param interlocutor_id: уникальный строковый идентификатор собеседника
        :return: класс, производный от BaseDialogSession
        """
        raise NotImplementedError()

    def prune(self):
        """ Удаляем сессии, в которых давно нет никакой активности """
        raise NotImplementedError()

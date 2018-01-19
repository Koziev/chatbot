# -*- coding: utf-8 -*-

from base_dialog_session import BaseDialogSession

class BaseDialogSessionFactory(object):
    """
    Базовый класс для управления диалоговыми сессиями - загрузка из
    хранилища, вопросы персистентности.
    """
    def __init__(self):
        pass

    def __getitem__(self, interlocutor):
        """
        Получение объекта сессии по идентификатору собеседника.
        При необходимости сессия может быть создана и загружена из хранилища.
        :param interlocutor: уникальный строковый идентификатор собеседника
        :return: класс, производный от BaseDialogSession
        """
        raise NotImplemented()

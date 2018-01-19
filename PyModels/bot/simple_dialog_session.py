# -*- coding: utf-8 -*-

from base_dialog_session import BaseDialogSession

class SimpleDialogSession(BaseDialogSession):
    """
    Простейшее хранилище диалоговой сессии.
    """
    def __init__(self, interlocutor, facts_storage):
        """
        Инициализация новой диалоговой сессии для нового собеседника.
        :param interlocutor: строковый идентификатор собеседника.
        """
        super(SimpleDialogSession, self).__init__(interlocutor, facts_storage)



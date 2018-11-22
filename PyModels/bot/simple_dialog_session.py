# -*- coding: utf-8 -*-

from base_dialog_session import BaseDialogSession

class SimpleDialogSession(BaseDialogSession):
    """
    Простейшее хранилище диалоговой сессии.
    """
    def __init__(self, bot_id, interlocutor, facts_storage):
        """
        Инициализация новой диалоговой сессии для нового собеседника.
        """
        super(SimpleDialogSession, self).__init__(bot_id, interlocutor, facts_storage)



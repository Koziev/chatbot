# -*- coding: utf-8 -*-

from bot.base_session_factory import BaseDialogSessionFactory
from bot.simple_dialog_session import SimpleDialogSession


class SimpleDialogSessionFactory(BaseDialogSessionFactory):
    """
    Простейшее хранилище сессий диалога между ботами и собеседниками.
    """
    def __init__(self):
        super(SimpleDialogSessionFactory, self).__init__()
        self.sessions = dict()

    def get_session(self, bot, interlocutor_id):
        assert(interlocutor_id is not None and len(interlocutor_id) != 0)
        assert(bot is not None)

        session_key = bot.get_bot_id() + '|' + interlocutor_id

        if session_key not in self.sessions:
            # Создаем новую сессию для этой пары бота и пользователя
            self.sessions[session_key] = SimpleDialogSession(bot.get_bot_id(), interlocutor_id, bot.facts)

        return self.sessions[session_key]


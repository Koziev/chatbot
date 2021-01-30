# -*- coding: utf-8 -*-

import logging
import datetime

from ruchatbot.bot.base_session_factory import BaseDialogSessionFactory
from ruchatbot.bot.simple_dialog_session import SimpleDialogSession


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

    def prune_session(self, session):
        l = logging.getLogger('SimpleDialogSessionFactory')
        l.info('prune_session bot=%s interlocutor=%s started=%s last_activity=%s', session.get_bot_id(), session.get_interlocutor(), session.get_start_time(), session.get_last_activity_time())

        self.logger.debug('============================= START OF PRUNING SESSION ============================')
        for i, item in enumerate(session.conversation_history):
            if item.is_bot_phrase:
                label = 'B'
            else:
                label = 'H'
            self.logger.debug('%2d| %s: - %s', i, label, item.raw_phrase)
        self.logger.debug('============================= END OF PRUNING SESSION ============================')

        del session

    def prune(self):
        active_sessions = dict()
        cur = datetime.datetime.now()
        for key, session in self.sessions.items():
            idle_time = (cur - session.get_last_activity_time()).total_seconds()
            if idle_time > 3600:
                self.prune_session(session)
            else:
                active_sessions[key] = session

        self.sessions = active_sessions

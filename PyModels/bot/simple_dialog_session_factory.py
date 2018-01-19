# -*- coding: utf-8 -*-

from base_session_factory import BaseDialogSessionFactory
from simple_dialog_session import SimpleDialogSession

class SimpleDialogSessionFactory(BaseDialogSessionFactory):
    """
    Простейшее хранилище.
    """
    def __init__(self, facts_storage):
        super(SimpleDialogSessionFactory, self).__init__()
        self.sessions = dict()
        self.facts_storage = facts_storage

    def __getitem__(self, interlocutor):
        assert(interlocutor is not None and len(interlocutor)!=0)
        if interlocutor not in self.sessions:
            # Создаем новую сессию для этого пользователя
            self.sessions[interlocutor] = SimpleDialogSession(interlocutor, self.facts_storage )

        return self.sessions[interlocutor]


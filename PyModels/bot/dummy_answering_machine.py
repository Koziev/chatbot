# -*- coding: utf-8 -*-

from base_answering_machine import BaseAnsweringMachine
from simple_dialog_session_factory import SimpleDialogSessionFactory
from null_facts_storage import NullFactsStorage

class DummyAnsweringMachine(BaseAnsweringMachine):
    """
    Класс, реализующий минимальный функционал чат-бота для
    проверки контрактов. Пустая база фактов, отсутствие запоминания
    новых фактов.
    """

    def __init__(self):
        super(DummyAnsweringMachine,self).__init__()
        self.session_factory = SimpleDialogSessionFactory(NullFactsStorage())

    def get_session_factory(self):
        return self.session_factory

    def push_phrase(self, bot, interlocutor, phrase):
        session = self.get_session(bot, interlocutor)
        session.add_to_buffer(phrase) # эхо - входная реплика копируется в ответ


    def pop_phrase(self, bot, interlocutor):
        session = self.get_session(bot, interlocutor)
        return session.extract_from_buffer()

    def get_session(self, interlocutor):
        return self.get_session_factory()[interlocutor]
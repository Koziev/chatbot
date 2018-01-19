# -*- coding: utf-8 -*-

from base_facts_storage import BaseFactsStorage

class NullFactsStorage(BaseFactsStorage):
    """
    Реализация всегда пустого хранилища фактов.
    """

    def __init__(self):
        pass

    def enumerate_facts(self, interlocutor):
        pass

    def store_new_fact(self, interlocutor, fact):
        pass

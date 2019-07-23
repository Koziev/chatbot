# -*- coding: utf-8 -*-

import os
import codecs
import datetime
import itertools

from smalltalk_rules import SmalltalkReplicas
from simple_facts_storage import SimpleFactsStorage

class OnlineFactsStorage(SimpleFactsStorage):
    """
    Факты задаются списком.
    Новые факты хранятся только в памяти.
    """

    def __init__(self, text_utils, predefined_facts):
        """
        :param text_utils: экземпляр класса TextUtils
        :param predefined_facts - список из кортежей (строка_факты, лицо, уникальный_id)
        """
        super(OnlineFactsStorage, self).__init__(text_utils)
        self.text_utils = text_utils
        self.predefined_facts = predefined_facts[:]
        self.new_facts = []

    def enumerate_facts(self, interlocutor):
        # родительский класс добавит факты о текущем времени и т.д.
        memory_phrases = list(super(OnlineFactsStorage, self).enumerate_facts(interlocutor))

        # Добавляем факты, загружаемые из текстовых файлов.
        for fact, person, fact_id in self.predefined_facts:
            canonized_line = self.text_utils.canonize_text(fact)
            memory_phrases.append((canonized_line, person, ''))

        for f in itertools.chain(self.new_facts, memory_phrases):
            yield f

    def store_new_fact(self, interlocutor, fact):
        self.new_facts.append(fact)

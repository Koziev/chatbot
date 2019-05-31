# -*- coding: utf-8 -*-

import os
import codecs
import itertools

from bot.simple_facts_storage import SimpleFactsStorage


class Files3FactsStorage(SimpleFactsStorage):
    """
    Класс читает факты из нескольких файлов, игнорируя идентификатор собеседника.
    Новые факты хранятся только в памяти, таким образом персистентность не
    реализована.
    """

    def __init__(self, text_utils, facts_folder):
        """
        :param text_utils: экземпляр класса TextUtils
        :param facts_folder: папка, в которой лежат текстовые файлы с фактами
        """
        super(Files3FactsStorage, self).__init__(text_utils)
        self.text_utils = text_utils
        self.facts_folder = facts_folder
        self.new_facts = []

    def enumerate_facts(self, interlocutor):
        premises_3_path = os.path.join(self.facts_folder, 'premises.txt')
        premises_1s_path = os.path.join(self.facts_folder, 'premises_1s.txt')
        premises_2s_path = os.path.join(self.facts_folder, 'premises_2s.txt')

        # родительский класс добавит факты о текущем времени и т.д.
        memory_phrases = list(super(Files3FactsStorage, self).enumerate_facts(interlocutor))

        # Добавляем факты, загружаемые из текстовых файлов.
        for p, ptype in [(premises_3_path, '3'), (premises_1s_path, '1s'), (premises_2s_path, '2s')]:
            with codecs.open(p, 'r', 'utf-8') as rdr:
                for line in rdr:
                    line1 = line.strip()
                    # Строки, начинающиеся на #, считаем комментариями и пропускаем.
                    if not line1.startswith('#') and len(line1) > 2:
                        canonized_line = self.text_utils.canonize_text(line1)
                        memory_phrases.append((canonized_line, ptype, ''))

        for f in itertools.chain(self.new_facts, memory_phrases):
            yield f

    def store_new_fact(self, interlocutor, fact):
        self.new_facts.append(fact)

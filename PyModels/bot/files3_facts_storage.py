# -*- coding: utf-8 -*-

import os
import codecs
import datetime
import itertools

from smalltalk_replicas import SmalltalkReplicas
from simple_facts_storage import SimpleFactsStorage

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
        super(self).__init__()
        self.text_utils = text_utils
        self.facts_folder = facts_folder
        self.new_facts = []

    def enumerate_smalltalk_replicas(self):
        """
        Из отдельного текстового файла загружается и возвращается список
        реплик в ответ на не-вопрос собеседника.
        :return: перечислимая последовательность экземпляров SmalltalkReplicas
        """
        smalltalk_path = os.path.join(self.facts_folder, 'smalltalk.txt')
        smalltalk_replicas = []
        with codecs.open(smalltalk_path, 'r', 'utf-8') as rdr:
            q_list = []
            a_list = []
            for iline, line in enumerate(rdr):
                line = line.strip()
                if len(line) == 0:
                    for q in q_list:
                        q = u' '.join(self.text_utils.tokenize(q))
                        item = SmalltalkReplicas(q)
                        for a in a_list:
                            a = a.strip()
                            item.add_answer(a)
                        assert(len(item.query) > 0)
                        if len(item.answers) == 0:
                            raise RuntimeError(u'Empty list of answers for smalltalk phrase \"{}\"'.format(item.query))
                        smalltalk_replicas.append(item)

                    q_list = []
                    a_list = []
                elif line.startswith('Q:'):
                    q_list.append(line.replace('Q:', '').strip())
                elif line.startswith('A:'):
                    a_list.append(line.replace('A:', '').strip())

        return smalltalk_replicas

    def enumerate_facts(self, interlocutor):
        premises_3_path = os.path.join(self.facts_folder, 'premises.txt')
        premises_1s_path = os.path.join(self.facts_folder, 'premises_1s.txt')
        premises_2s_path = os.path.join(self.facts_folder, 'premises_2s.txt')

        # родительский класс добавит факты о текущем времени и т.д.
        memory_phrases = list(super(self).enumerate_facts(interlocutor))

        # Добавляем факты, загружаемые из текстовых файлов.
        for p, ptype in [(premises_3_path, '3'), (premises_1s_path, '1s'), (premises_2s_path, '2s')]:
            with codecs.open(p, 'r', 'utf-8') as rdr:
                for line in rdr:
                    line1 = line.strip()
                    if len(line1)>2:
                        canonized_line = self.text_utils.canonize_text(line1)
                        memory_phrases.append((canonized_line, ptype, ''))

        for f in itertools.chain(self.new_facts, memory_phrases):
            yield f

    def store_new_fact(self, interlocutor, fact):
        self.new_facts.append(fact)

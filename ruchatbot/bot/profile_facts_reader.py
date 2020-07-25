# -*- coding: utf-8 -*-

import io
import itertools
import logging

from ruchatbot.bot.simple_facts_storage import SimpleFactsStorage
from ruchatbot.utils.constant_replacer import replace_constant


class ProfileFactsReader(SimpleFactsStorage):
    """
    Класс читает факты из одного файла. Новые факты (например, имя собеседника) хранятся только в памяти,
    таким образом персистентность не реализована.
    """

    def __init__(self, text_utils, profile_path, constants):
        """
        :param text_utils: экземпляр класса TextUtils
        :param profile_path: путь к текстовому файлу с фактами
        """
        super(ProfileFactsReader, self).__init__(text_utils)
        self.text_utils = text_utils
        self.profile_path = profile_path
        self.profile_facts = None
        self.constants = constants
        self.new_facts = []

    def load_profile(self):
        logger = logging.getLogger('ProfileFactsReader')
        if self.profile_facts is None:
            logger.info(u'Loading profile facts from "%s"', self.profile_path)
            self.profile_facts = []
            with io.open(self.profile_path, 'r', encoding='utf=8') as rdr:
                current_section = None
                for line in rdr:
                    line = line.strip()
                    if line:
                        if line.startswith('#'):
                            if line.startswith('##'):
                                current_section = line[line.index(':')+1:].strip()
                                if current_section not in ('1s', '2s', '3'):
                                    msg = u'Unknown profile section {}'.format(current_section)
                                    raise RuntimeError(msg)
                            else:
                                # Строки с одним # считаем комментариями.
                                continue
                        else:
                            assert(current_section)
                            canonized_line = self.text_utils.canonize_text(line)
                            canonized_line = replace_constant(canonized_line, self.constants, self.text_utils)
                            self.profile_facts.append((canonized_line, current_section, u''))
            logger.debug(u'%d facts loaded from "%s"', len(self.profile_facts), self.profile_path)

    def reset_added_facts(self):
        self.new_facts = []

    def enumerate_facts(self, interlocutor):
        # Загрузим факты из профиля, если еще не загрузили.
        self.load_profile()

        # родительский класс добавит факты о текущем времени и т.д.
        parent_facts = list(super(ProfileFactsReader, self).enumerate_facts(interlocutor))

        for f in itertools.chain(self.new_facts, self.profile_facts, parent_facts):
            yield f

    def store_new_fact(self, interlocutor, fact, unique):
        # Новые факты, добавляемые собеседником в ходе диалога, сохраняем только в оперативке.
        if unique:
            # Ищем факт с именем fact[2], если найден - заменяем, а не вносим новый.
            found = False
            for i, fact0 in enumerate(self.new_facts):
                if fact0[2] == fact[2]:
                    self.new_facts[i] = fact
                    found = True
                    break

            if not found:
                self.new_facts.append(fact)
        else:
            self.new_facts.append(fact)

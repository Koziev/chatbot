# -*- coding: utf-8 -*-
"""
Реализация хранилища фактов (Базы Знаний) в обычном текстовом файле, без персистентности новых фактов.

30.12.2020 Добавляем возможность догрузки фактов из других файлов с помощью директивы "## import XXX",
           чтобы общие для нескольких профилей факты хранить в одной файле.

02.01.2020 Храним новые факты в привязке к идентификатору собеседника, чтобы нормально работал режим параллельного
           диалога с множеством собеседников.

29.01.2021 Генерируемые факты - при чтении строки из профиля она разбивается по символу | и выбирается одна из
           получившихся строк. Таким образом можно вводить вариативность в набор фактов.
"""

import io
import itertools
import os
import re
import logging
import random
import collections

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
        :param profile_path: путь к основному текстовому файлу с фактами
        """
        super(ProfileFactsReader, self).__init__(text_utils)
        self.text_utils = text_utils
        self.profile_path = profile_path
        self.profile_facts = None
        self.constants = constants
        self.new_facts = collections.defaultdict(list)  # списки новых фактов в привязке к id собеса
        self.logger = logging.getLogger('ProfileFactsReader')

    def load_profile(self):
        if self.profile_facts is None:
            self.logger.info('Loading profile facts from "%s"', self.profile_path)
            self.profile_facts = []
            if self.profile_path is not None:
                with io.open(self.profile_path, 'r', encoding='utf=8') as rdr:
                    current_section = None
                    for line in rdr:
                        line = line.strip()
                        if line:
                            if line.startswith('#'):
                                if line.startswith('##'):
                                    if 'profile_section:' in line:
                                        # Задается раздел баз знаний
                                        current_section = line[line.index(':')+1:].strip()
                                        if current_section not in ('1s', '2s', '3'):
                                            msg = 'Unknown profile section {}'.format(current_section)
                                            raise RuntimeError(msg)
                                    elif 'import' in line:
                                        # Читаем факты из дополнительного файла
                                        fn = re.search('import "(.+)"', line).group(1).strip()
                                        add_path = os.path.join(os.path.dirname(self.profile_path), fn)
                                        self.logger.debug('Loading facts from file "%s"...', add_path)
                                        with io.open(add_path, 'rt', encoding='utf-8') as rdr2:
                                            for line in rdr2:
                                                line = line.strip()
                                                if line and not line.startswith('#'):
                                                    line1 = random.choice(line.split('|')).strip()
                                                    canonized_line = self.text_utils.canonize_text(line1)
                                                    canonized_line = replace_constant(canonized_line, self.constants, self.text_utils)
                                                    self.profile_facts.append((canonized_line, current_section, add_path))

                                else:
                                    # Строки с одним # считаем комментариями.
                                    continue
                            else:
                                assert(current_section)
                                line1 = random.choice(line.split('|')).strip()
                                canonized_line = self.text_utils.canonize_text(line1)
                                canonized_line = replace_constant(canonized_line, self.constants, self.text_utils)
                                self.profile_facts.append((canonized_line, current_section, self.profile_path))
            self.logger.debug('%d facts loaded from "%s"', len(self.profile_facts), self.profile_path)

    def reset_added_facts(self):
        self.new_facts = collections.defaultdict(list)

    def reset_all_facts(self):
        self.reset_added_facts()
        self.profile_facts = None

    def enumerate_facts(self, interlocutor):
        # Загрузим факты из профиля, если еще не загрузили.
        self.load_profile()

        # родительский класс добавит факты о текущем времени и т.д.
        parent_facts = list(super(ProfileFactsReader, self).enumerate_facts(interlocutor))

        for f in itertools.chain(self.new_facts[interlocutor], self.profile_facts, parent_facts):
            yield f

    def store_new_fact(self, interlocutor, fact, unique):
        if fact[0].count(' ') == 0:
            self.logger.error('1-word facts are not valid!: interlocutor=%s fact=%s', interlocutor, fact[0])
            return

        # Новые факты, добавляемые собеседником в ходе диалога, сохраняем только в оперативке,
        # в других реализациях хранилища будет персистентность.
        interlocutor_facts = self.new_facts[interlocutor]
        if unique:
            # Ищем факт с именем fact[2], если найден - заменяем, а не вносим новый.
            found = False
            for i, fact0 in enumerate(interlocutor_facts):
                if fact0[2] == fact[2]:
                    interlocutor_facts[i] = fact
                    found = True
                    break

            if not found:
                interlocutor_facts.append(fact)
        else:
            interlocutor_facts.append(fact)

    def get_added_facts(self, interlocutor):
        return self.new_facts[interlocutor]

    def find_tagged_fact(self, interlocutor, fact_tag):
        """Среди новых фактов ищем имеющий указанный тэг"""
        for fact, section, tag in self.new_facts[interlocutor]:
            if tag == fact_tag:
                return fact

        return None


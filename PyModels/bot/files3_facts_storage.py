# -*- coding: utf-8 -*-

from base_facts_storage import BaseFactsStorage

import os
import codecs
import datetime
import itertools

class Files3FactsStorage(BaseFactsStorage):
    """
    Класс читает факты из нескольких файлов, игнорируя идентификатор собеседника.
    Новые факты храняться только в памяти, таким образом персистентность не
    реализована.
    """

    def __init__(self, text_utils, facts_folder):
        """
        :param text_utils: экземпляр класса TextUtils
        :param facts_folder: папка, в которой лежат текстовые файлы с фактами
        """
        super(Files3FactsStorage, self).__init__()
        self.text_utils = text_utils
        self.facts_folder = facts_folder
        self.new_facts = []

    def enumerate_facts(self, interlocutor):
        premises_3_path = os.path.join(self.facts_folder, 'premises.txt')
        premises_1s_path = os.path.join(self.facts_folder, 'premises_1s.txt')
        premises_2s_path = os.path.join(self.facts_folder, 'premises_2s.txt')

        memory_phrases = [] # TODO: возвращать факты сразу по мере чтения из файла через yield
        for p, ptype in [(premises_3_path, '3'), (premises_1s_path, '1s'), (premises_2s_path, '2s')]:
            with codecs.open(p, 'r', 'utf-8') as rdr:
                for line in rdr:
                    line1 = line.strip()
                    if len(line1)>2:
                        canonized_line = self.text_utils.canonize_text(line1)
                        memory_phrases.append( (canonized_line, ptype, '') )

        # Добавляем текущие факты

        # День недели
        dw = [u'понедельник', u'вторник', u'среда', u'четверг',
              u'пятница', u'суббота', u'воскресенье'][datetime.datetime.today().weekday()]

        memory_phrases.append( (u'сегодня '+dw, '3', 'current_date') )

        # Время года
        #currentSecond= datetime.now().second
        #currentMinute = datetime.now().minute
        #currentHour = datetime.now().hour

        #currentDay = datetime.now().day
        cur_month = datetime.datetime.now().month
        #currentYear = datetime.now().year

        season = {11:u'зима', 12:u'зима', 1:u'зима',
                  2:u'весна', 3:u'весна', 4:u'весна',
                  5:u'лето', 6:u'лето', 7:u'лето',
                  8:u'осень', 9:u'осень', 10:u'осень'}[cur_month]
        memory_phrases.append( (u'сейчас '+season, '3', 'current_season') )

        # Добавляем текущее время с точностью до минуты
        current_minute = datetime.datetime.now().minute
        current_hour = datetime.datetime.now().hour
        current_time = u'Сейчас ' + str(current_hour)
        if (current_hour % 10) == 1:
            current_time += u' час '
        elif (current_hour % 10) in [2, 3, 4]:
            current_time += u' часа '
        else:
            current_time += u' часов '

        current_time += str(current_minute)
        if (current_minute % 10) == 1:
            current_time += u' минута '
        elif (current_minute % 10) in [2, 3, 4]:
            current_time += u' минуты '
        else:
            current_time += u' минут '

        memory_phrases.append((current_time, '3', 'current_time'))

        # возвращаем список фактов (потом надо переделать на выдачу по мере чтения из файла и
        # генерации через yield).
        for f in itertools.chain(self.new_facts, memory_phrases):
            yield f

    def store_new_fact(self, interlocutor, fact):
        self.new_facts.append(fact)

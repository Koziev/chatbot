# -*- coding: utf-8 -*-
"""
29.06.2020 Добавлены динамические факты "current_day_month" со строкой типа "сегодня 29 июня" и
           "current_year" со строкой типа "сейчас 2020 год"
"""


from ruchatbot.bot.base_facts_storage import BaseFactsStorage

import datetime
import itertools


class SimpleFactsStorage(BaseFactsStorage):
    """
    Базовый класс хранилища фактов с добавленной функциональностью:
    1) метод enumerate_smalltalk_replicas возвращает пустой список, а не
       бросает исключение.
    2) факты о текущем времени и дне недели добавляются в список, возвращаемый
       методом enumerate_facts.
    """

    def __init__(self, text_utils):
        """
        :param text_utils: экземпляр класса TextUtils
        """
        super(SimpleFactsStorage, self).__init__()
        self.text_utils = text_utils

    def enumerate_smalltalk_replicas(self):
        return []

    def enumerate_facts(self, interlocutor):
        memory_phrases = []

        # Добавляем динамические факты
        # ==== День недели ====
        dwos = 'понедельник вторник среда четверг пятница суббота воскресенье'.split()
        today = datetime.datetime.today()
        s = dwos[today.weekday()]
        memory_phrases.append(('сегодня ' + s, '3', 'current_day_of_week'))

        yesterday = today - datetime.timedelta(days=1)
        s = dwos[yesterday.weekday()]
        if s[-1] in 'кг':
            memory_phrases.append(('вчера был ' + s, '3', 'yesterday_day_of_week'))
        elif s[-1] == 'е':
            memory_phrases.append(('вчера было ' + s, '3', 'yesterday_day_of_week'))
        else:
            memory_phrases.append(('вчера была ' + s, '3', 'yesterday_day_of_week'))

        tomorrow = today + datetime.timedelta(days=1)
        s = dwos[tomorrow.weekday()]
        memory_phrases.append(('завтра будет ' + s, '3', 'tomorrow_day_of_week'))


        # === Время года ===
        cur_month = datetime.datetime.now().month
        season = {12: u'зима', 1: u'зима', 2: u'зима',
                  3: u'весна', 4: u'весна', 5: u'весна',
                  6: u'лето', 7: u'лето', 8: u'лето',
                  9: u'осень', 10: u'осень', 11: u'осень'}[cur_month]
        memory_phrases.append((u'сейчас ' + season, '3', 'current_season'))

        # === Текущий месяц ===
        month = {1: u'январь', 2: u'февраль', 3: u'март',
                 4: u'апрель', 5: u'май', 6: u'июнь', 7: u'июль',
                 8: u'август', 9: u'сентябрь', 10: u'октябрь', 11: u'ноябрь', 12: u'декабрь'}[cur_month]
        memory_phrases.append((u'сейчас ' + month, '3', 'current_month'))

        # Добавляем текущее время с точностью до минуты
        current_minute = datetime.datetime.now().minute
        current_hour = datetime.datetime.now().hour
        current_time = u'Сейчас ' + str(current_hour)
        if 20 >= current_hour >= 5:
            current_time += u' часов '
        elif current_hour in [1, 21]:
            current_time += u' час '
        elif (current_hour % 10) in [2, 3, 4]:
            current_time += u' часа '
        else:
            current_time += u' часов '

        current_time += str(current_minute)
        if current_minute > 11 and (current_minute % 10) == 1:
            current_time += u' минута '
        elif current_minute > 4 and (current_minute % 10) in [2, 3, 4]:
            current_time += u' минуты '
        else:
            current_time += u' минут '

        memory_phrases.append((current_time, '3', 'current_time'))

        # Текущая дата в формате "29 июня"
        cur_day = datetime.datetime.now().day
        month_gen = {1: u'января', 2: u'февраля', 3: u'марта',
                     4: u'апреля', 5: u'мая', 6: u'июня', 7: u'июля',
                     8: u'августа', 9: u'сентября', 10: u'октября', 11: u'ноября', 12: u'декабря'}[cur_month]
        memory_phrases.append(('сегодня {} {}'.format(cur_day, month_gen), '3', 'current_day_month'))

        # Текущий год
        memory_phrases.append(('сейчас {} год'.format(today.year), '3', 'current_year'))

        # возвращаем список фактов (потом надо переделать на выдачу по мере чтения из файла и
        # генерации через yield).
        for f in itertools.chain(self.new_facts, memory_phrases):
            yield f

    def store_new_fact(self, interlocutor, fact):
        pass

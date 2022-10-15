# -*- coding: utf-8 -*-
"""
29.06.2020 Добавлены динамические факты "current_day_month" со строкой типа "сегодня 29 июня" и
           "current_year" со строкой типа "сейчас 2020 год"
03.03.2022 Добавлены динамические факты "сейчас утро|день|вечер|ночь"
"""


from ruchatbot.bot.base_facts_storage import BaseFactsStorage

import datetime
import itertools


class SimpleFactsStorage(BaseFactsStorage):
    """
    Базовый класс хранилища фактов с добавленной функциональностью:
    1) факты о текущем времени и дне недели добавляются в список, возвращаемый
       методом enumerate_facts.
    """

    def __init__(self):
        super(SimpleFactsStorage, self).__init__()

    def reset_added_facts(self):
        pass

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

        # 03.03.2022 Часть суток
        current_hour = datetime.datetime.now().hour
        if current_hour >= 23 or current_hour < 6:
            tod_fact = 'сейчас ночь.'
        elif current_hour in [6, 7, 8, 9]:
            tod_fact = 'сейчас утро.'
        elif current_hour in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
            tod_fact = 'сейчас день.'
        else:
            tod_fact = 'сейчас вечер.'
        memory_phrases.append((tod_fact, '3', 'current_times_of_day'))

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
            current_time += ' минут '

        memory_phrases.append((current_time, '3', 'current_time'))

        # Текущая дата в формате "29 июня"
        cur_day = datetime.datetime.now().day
        month_gen = {1: 'января',  2: 'февраля', 3: 'марта',
                     4: 'апреля',  5: 'мая',     6: 'июня', 7: 'июля',
                     8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}[cur_month]
        memory_phrases.append(('сегодня {} {}'.format(cur_day, month_gen), '3', 'current_day_month'))

        # Текущий год
        memory_phrases.append(('сейчас идет {} год'.format(today.year), '3', 'current_year'))

        # возвращаем список фактов (потом надо переделать на выдачу по мере чтения из файла и
        # генерации через yield).
        #for f in memory_phrases:
        #    yield f
        return memory_phrases

    def store_new_fact(self, interlocutor, fact, unique):
        raise NotImplementedError()

    def find_tagged_fact(self, interlocutor, fact_tag):
        raise NotImplementedError()

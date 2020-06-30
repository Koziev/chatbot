# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для диалогового движка (https://github.com/Koziev/chatbot).
23.06.2020 добавлен пакетный режим как временный эрзац пакетного теста
"""

import os
import argparse
import logging
import io

from ruchatbot.bot.bot_profile import BotProfile
from ruchatbot.bot.profile_facts_reader import ProfileFactsReader
from ruchatbot.bot.text_utils import TextUtils
from ruchatbot.bot.simple_answering_machine import SimpleAnsweringMachine
from ruchatbot.bot.console_utils import input_kbd, print_answer, print_tech_banner, flush_logging
from ruchatbot.bot.bot_scripting import BotScripting
from ruchatbot.bot.bot_personality import BotPersonality
from ruchatbot.bot.plain_file_faq_storage import PlainFileFaqStorage
from ruchatbot.utils.logging_helpers import init_trainer_logging

from ruchatbot.scenarios.scenario_who_am_i import Scenario_WhoAmI


def on_order(order_anchor_str, bot, session):
    bot.say(session, u'Выполняю команду \"{}\"'.format(order_anchor_str))
    # Всегда возвращаем True, как будто можем выполнить любой приказ.
    # В реальных сценариях нужно вернуть False, если приказ не опознан
    return True


def on_weather_forecast(bot, session, user_id, interpreted_phrase, verb_form_fields):
    """
    Обработчик запросов для прогноза погоды.
    Вызывается ядром чатбота.
    :return: текст ответа, который увидит пользователь
    """
    when_arg = bot.extract_entity(u'когда', interpreted_phrase)
    return u'Прогноз погоды на момент времени "{}" сгенерирован в функции on_weather_forecast для демонстрации'.format(when_arg)


def on_check_emails(bot, session, user_id, interpreted_phrase, verb_form_fields):
    """
    Обработчик запросов на проверку электронной почты (реплики типа "Нет ли новых писем?")
    """
    return u'Фиктивная проверка почты в функции on_check_email'


def on_alarm_clock(bot, session, user_id, interpreted_phrase, verb_form_fields):
    when_arg = bot.extract_entity(u'когда', interpreted_phrase)
    return u'Фиктивный будильник для "{}"'.format(when_arg)


def on_buy_pizza(bot, session, user_id, interpreted_phrase, verb_form_fields):
    if interpreted_phrase:
        meal_arg = bot.extract_entity(u'объект', interpreted_phrase)
        count_arg = bot.extract_entity(u'количество', interpreted_phrase)
    else:
        meal_arg = verb_form_fields['что_заказывать']
        count_arg = verb_form_fields['количество_порций']

    return u'Заказываю: что="{}", сколько="{}"'.format(meal_arg, count_arg)


def main():
    user_id = 'test'

    parser = argparse.ArgumentParser(description='Question answering machine')
    parser.add_argument('--data_folder', type=str, default='../../data')
    parser.add_argument('--w2v_folder', type=str, default='../../tmp')
    parser.add_argument('--profile', type=str, default='../../data/profile_1.json', help='path to profile file')
    parser.add_argument('--models_folder', type=str, default='../../tmp', help='path to folder with pretrained models')
    parser.add_argument('--tmp_folder', type=str, default='../../tmp', help='path to folder for logfile etc')
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--input', type=str, help='Input file with phrases')
    parser.add_argument('--output', type=str, help='Output file with bot replicas')

    args = parser.parse_args()
    profile_path = os.path.expanduser(args.profile)
    models_folder = os.path.expanduser(args.models_folder)
    data_folder = os.path.expanduser(args.data_folder)
    w2v_folder = os.path.expanduser(args.w2v_folder)
    tmp_folder = os.path.expanduser(args.tmp_folder)

    init_trainer_logging(os.path.join(tmp_folder, 'console_chatbot.log'), args.debugging)

    logging.debug('Bot loading...')

    # Создаем необходимое окружение для бота.

    # NLP pileline: содержит инструменты для работы с текстом, включая морфологию и таблицы словоформ,
    # part-of-speech tagger, NP chunker и прочее.
    text_utils = TextUtils()
    text_utils.load_embeddings(w2v_dir=w2v_folder, wc2v_dir=models_folder)
    text_utils.load_dictionaries(data_folder, models_folder)

    # Настроечные параметры аватара собраны в профиле - файле в json формате.
    profile = BotProfile()
    profile.load(profile_path, data_folder, models_folder)

    # Инициализируем движок вопросно-ответной системы. Он может обслуживать несколько
    # ботов с разными провилями (базами фактов и правил), хотя тут у нас будет работать только один.
    machine = SimpleAnsweringMachine(text_utils=text_utils)
    machine.load_models(data_folder, models_folder, profile.constants)
    machine.trace_enabled = args.debugging

    # Контейнер для правил
    scripting = BotScripting(data_folder)
    scripting.load_rules(profile.rules_path, profile.smalltalk_generative_rules, profile.constants, text_utils)

    # Добавляем скрипты на питоне
    scripting.add_scenario(Scenario_WhoAmI())

    # Конкретная реализация хранилища фактов - плоские файлы в utf-8, с минимальным форматированием
    profile_facts = ProfileFactsReader(text_utils=text_utils,
                                       profile_path=profile.premises_path,
                                       constants=profile.constants)

    # Подключем простое файловое хранилище с FAQ-правилами бота.
    # Движок бота сопоставляет вопрос пользователя с опорными вопросами в FAQ базе,
    # и если нашел хорошее соответствие (синонимичность выше порога), то
    # выдает ответную часть найденной записи.
    faq_storage = PlainFileFaqStorage(profile.faq_path, constants=profile.constants, text_utils=text_utils)

    # Инициализируем аватара
    bot = BotPersonality(bot_id='test_bot',
                         engine=machine,
                         facts=profile_facts,
                         faq=faq_storage,
                         scripting=scripting,
                         profile=profile)

    bot.on_process_order = on_order

    # Выполняем привязку обработчиков
    bot.add_event_handler(u'weather_forecast', on_weather_forecast)
    bot.add_event_handler(u'check_emails', on_check_emails)
    bot.add_event_handler(u'alarm_clock', on_alarm_clock)
    bot.add_event_handler(u'buy_pizza', on_buy_pizza)

    bot.start_conversation(user_id)
    flush_logging()
    print_tech_banner()

    if args.input:
        # Пакетный режим - читаем фразы собеседника из указанного текстового файла, прогоняем
        # через бота, сохраняем ответные фразы бота в выходной файл.
        with io.open(args.input, 'r', encoding='utf-8') as rdr,\
             io.open(args.output, 'w', encoding='utf-8') as wrt:
            for line in rdr:
                inline = line.strip()
                if inline.startswith('#'):
                    # комментарии просто сохраняем в выходном файле для
                    # удобства визуальной организации
                    wrt.write('\n{}\n'.format(inline))
                    continue

                if inline:
                    wrt.write('\nH: {}\n'.format(inline))
                    bot.push_phrase(user_id, line.strip())

                    while True:
                        answer = bot.pop_phrase(user_id)
                        if len(answer) == 0:
                            break
                        wrt.write('B: {}\n'.format(answer))
    else:
        # Консольный интерактивный режим
        while True:
            print('\n')

            # В самом начале диалога, когда еще не было ни одной реплики,
            # бот может сгенерировать некое приветствие или вопрос для
            # завязывания беседы. Поэтому сразу извлечем сгенерированные фразы из
            # буфера и покажем их.
            while True:
                answer = bot.pop_phrase(user_id)
                if len(answer) == 0:
                    break

                print_answer(u'B:>', answer)

            question = input_kbd('H:>')
            if len(question) > 0:
                if question.lower() in ('r\exit', r'\q', r'\quit', '/stop'):
                    break

                bot.push_phrase(user_id, question)


if __name__ == '__main__':
    main()

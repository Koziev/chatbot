# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для диалогового движка (https://github.com/Koziev/chatbot).
23.06.2020 добавлен пакетный режим как временный эрзац пакетного теста
16.07.2020 создание экземпляра бота с дефолтным функционалом вынесено в хелпер-функцию create_chatbot
16.07.2020 пакетный режим перенесен в chatbot_tester.py
11.11.2020 добавлен параметр: урл веб-сервиса чит-чата
"""

import os
import argparse
import logging
import io

from ruchatbot.bot.console_utils import input_kbd, print_answer, print_tech_banner, flush_logging
from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.frontend.bot_creator import create_chatbot, ChitchatConfig


def on_order(order_anchor_str, bot, session):
    if True:
        logging.debug('Order callback with order_anchor_str="%s"', order_anchor_str)
        return False
    else:
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


if __name__ == '__main__':
    user_id = 'anonymous'

    parser = argparse.ArgumentParser(description='Question answering machine')
    parser.add_argument('--data_folder', type=str, default='../../data')
    parser.add_argument('--w2v_folder', type=str, default='../../tmp')
    parser.add_argument('--profile', type=str, default='../../data/profile_1.json', help='path to profile file')
    parser.add_argument('--models_folder', type=str, default='../../tmp', help='path to folder with pretrained models')
    parser.add_argument('--tmp_folder', type=str, default='../../tmp', help='path to folder for logfile etc')
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--greeting', type=int, default=1)
    parser.add_argument('--chitchat_url', type=str, help='chit-chat service endpoint')

    args = parser.parse_args()
    profile_path = os.path.expanduser(args.profile)
    models_folder = os.path.expanduser(args.models_folder)
    data_folder = os.path.expanduser(args.data_folder)
    w2v_folder = os.path.expanduser(args.w2v_folder)
    tmp_folder = os.path.expanduser(args.tmp_folder)

    init_trainer_logging(os.path.join(tmp_folder, 'console_chatbot.log'), args.debugging)

    if args.chitchat_url:
        rugpt_chitchat_config = ChitchatConfig()
        rugpt_chitchat_config.service_endpoint = args.chitchat_url
        # rugpt_chitchat_config.temperature = 0.9
        rugpt_chitchat_config.num_return_sequences = 2
    else:
        rugpt_chitchat_config = None

    logging.debug('Bot loading...')
    bot = create_chatbot(profile_path, models_folder, w2v_folder, data_folder, args.debugging, chitchat_config=rugpt_chitchat_config)

    # Выполняем привязку обработчиков
    bot.on_process_order = on_order
    bot.add_event_handler(u'weather_forecast', on_weather_forecast)
    bot.add_event_handler(u'check_emails', on_check_emails)
    bot.add_event_handler(u'alarm_clock', on_alarm_clock)
    bot.add_event_handler(u'buy_pizza', on_buy_pizza)

    if args.greeting:
        bot.start_conversation(user_id)

    flush_logging()
    print_tech_banner()

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
            if question.lower() in (r'\exit', r'\q', r'\quit', '/stop'):
                break

        # 25-07-2020 пустая фраза имитирует ситуацию тупика в диалоге, бот должен попытаться
        # предложить продолжение...
        bot.push_phrase(user_id, question)

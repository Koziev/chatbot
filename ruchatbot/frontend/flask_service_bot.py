# -*- coding: utf-8 -*-
"""
Простой web api для чатбота https://github.com/Koziev/chatbot на Flask.
"""

from __future__ import print_function

import argparse
import os
import logging

from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.bot_service import flask_app
from ruchatbot.bot_service.global_params import profile_path, models_folder, data_folder, w2v_folder
from ruchatbot.bot.bot_profile import BotProfile
from ruchatbot.bot.plain_file_faq_storage import PlainFileFaqStorage
from ruchatbot.bot.profile_facts_reader import ProfileFactsReader
from ruchatbot.bot.text_utils import TextUtils
from ruchatbot.bot.simple_answering_machine import SimpleAnsweringMachine
from ruchatbot.bot.bot_scripting import BotScripting
from ruchatbot.bot.bot_personality import BotPersonality
from ruchatbot.scenarios.scenario_who_am_i import Scenario_WhoAmI


listen_ip = '127.0.0.1'
listen_port = 9001


@flask_app.before_first_request
def init_chatbot():
    if 'bot' not in flask_app.config:
        logging.info('init_chatbot: models_folder="%s"', models_folder)

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
        machine.trace_enabled = True  # включаем расширенную трассировку работы движка

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
        bot = BotPersonality(bot_id='flask_bot',
                             engine=machine,
                             facts=profile_facts,
                             faq=faq_storage,
                             scripting=scripting,
                             profile=profile)

        def on_order(order_anchor_str, bot, session):
            bot.say(session, 'Выполняю команду "{}"'.format(order_anchor_str))

        bot.on_process_order = on_order

        flask_app.config['bot'] = bot
        logging.info('init_chatbot complete')


if __name__ == '__main__':
    # Разбор параметров запуска, указанных в командной строке
    parser = argparse.ArgumentParser(description='Chatbot web-service')
    parser.add_argument('--profile', type=str, default='../../../data/profile_1.json', help='path to profile file')
    parser.add_argument('--data_folder', type=str, default='../../../data')
    parser.add_argument('--tmp_folder', type=str, default='../../../tmp')
    parser.add_argument('--w2v_folder', type=str, default='../../../tmp')
    parser.add_argument('--models_folder', type=str, default='../../../tmp', help='path to folder with pretrained models')
    parser.add_argument('--ip', help='listen to specified IP address', type=str, default=listen_ip)
    parser.add_argument('--port', type=str, default=listen_port)
    parser.add_argument('--preload', type=bool, default=True, help='Load all models and dictionary before service start-up')

    args = parser.parse_args()
    profile_path = os.path.expanduser(args.profile)
    models_folder = os.path.expanduser(args.models_folder)
    data_folder = os.path.expanduser(args.data_folder)
    w2v_folder = os.path.expanduser(args.w2v_folder)
    tmp_folder = os.path.expanduser(args.tmp_folder)
    listen_ip = args.ip
    listen_port = args.port

    # настраиваем логирование в файл
    init_trainer_logging(os.path.join(tmp_folder, 'flask_service_bot.log'), debugging=True)

    if args.preload:
        init_chatbot()

    logging.info('Going to run flask_app listening %s:%d profile_path="%s" models_folder="%s" data_folder="%s" w2v_folder="%s"', listen_ip, listen_port, profile_path, models_folder, data_folder, w2v_folder)
    flask_app.run(debug=True, host=listen_ip, port=listen_port)


# https://stackoverflow.com/questions/8495367/using-additional-command-line-arguments-with-gunicorn?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# Gunicorn entry point generator
def start_extractor(*args, **kwargs):
    global profile_path, models_folder, data_folder, w2v_folder, tmp_folder

    # Gunicorn CLI args are useless.
    # https://stackoverflow.com/questions/8495367/
    #
    # Start the application in modified environment.
    # https://stackoverflow.com/questions/18668947/
    #
    for k in kwargs:
        if k == 'profile_path':
            profile_path = kwargs[k]
        elif k == 'models_folder':
            models_folder = kwargs[k]
        elif k == 'data_folder':
            data_folder = kwargs[k]
        elif k == 'w2v_folder':
            w2v_folder = kwargs[k]
        elif k == 'tmp_folder':
            tmp_folder = kwargs[k]

    return flask_app

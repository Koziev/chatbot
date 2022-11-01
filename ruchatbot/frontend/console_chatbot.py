"""
Консольный фронтэнд для диалогового движка (https://github.com/Koziev/chatbot).
23.06.2020 добавлен пакетный режим как временный эрзац пакетного теста
16.07.2020 создание экземпляра бота с дефолтным функционалом вынесено в хелпер-функцию create_chatbot
16.07.2020 пакетный режим перенесен в chatbot_tester.py
11.11.2020 добавлен параметр: урл веб-сервиса чит-чата
01.22.2022 переделка на новую версию диалоговой системы
"""

import sys
import os
import argparse
import logging.handlers
import json

from ruchatbot.bot.conversation_engine import BotCore, SessionFactory
from ruchatbot.bot.text_utils import TextUtils
from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.bot.bot_profile import BotProfile
from ruchatbot.bot.facts_database import FactsDatabase


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Console UI for Сhatbot v7')
    parser.add_argument('--chatbot_dir', type=str, default=os.path.expanduser('~/polygon/chatbot'))
    parser.add_argument('--log', type=str, default=os.path.expanduser('~/polygon/chatbot/tmp/conversation_engine.log'))
    parser.add_argument('--profile', type=str, default=os.path.expanduser('~/polygon/chatbot/data/profile_1.json'), help='Path to yaml file with bot persona records')
    parser.add_argument('--bert', type=str)
    parser.add_argument('--db', type=str, default=':memory:', help='Connection string for SQLite storage file; use :memory: for no persistence')

    args = parser.parse_args()

    chatbot_dir = args.chatbot_dir
    models_dir = os.path.join(chatbot_dir, 'tmp')
    data_dir = os.path.join(chatbot_dir, 'data')
    tmp_dir = os.path.join(chatbot_dir, 'tmp')
    profile_path = args.profile

    init_trainer_logging(args.log, True)

    # Настроечные параметры бота собраны в профиле - файле в json формате.
    bot_profile = BotProfile("bot_v7")
    bot_profile.load(profile_path, data_dir, models_dir)

    text_utils = TextUtils()
    text_utils.load_dictionaries(data_dir, models_dir)

    # 19-03-2022 запрещаем тензорфлоу резервировать всю память в гпу по дефолту, так как
    # это мешает потом нормально работать моделям на торче.
    #for gpu in tf.config.experimental.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(gpu, True)

    bot = BotCore()
    if args.bert is not None:
        bot.load_bert(args.bert)
    else:
        # Определяем модель берта по конфигу детектора полных предпосылок.
        with open(os.path.join(models_dir, 'closure_detector_2.cfg'), 'r') as f:
            cfg = json.load(f)
            bert_name = cfg['bert_model']
            bot.load_bert(bert_name)

    bot.load(models_dir, text_utils)

    # Хранилище новых фактов, извлекаемых из диалоговых сессий.
    # По умолчанию размещается только в оперативной памяти.
    # Если в командной строке задана опция --db XXX, то будет использоваться БД sqlite для хранения новых фактов.
    facts_db = FactsDatabase(args.db)

    # Фабрика для создания новых диалоговых сессий и хранения текущих сессий для всех онлайн-собеседников
    session_factory = SessionFactory(bot_profile, text_utils, facts_db)

    # Консольный интерактивный режим для отладки.
    interlocutor = 'test_human'
    session = session_factory.start_conversation(interlocutor)

    # Начинаем со сценария приветствия.
    bot.start_greeting_scenario(session)

    while True:
        for handler in logging.getLogger().handlers:
            handler.flush()
        sys.stdout.flush()

        print('\n'.join(session.dialog.get_printable()), flush=True)
        h = input('H: ').strip()
        if h:
            session.dialog.add_human_message(h)
            replies = bot.process_human_message(session)
        else:
            break

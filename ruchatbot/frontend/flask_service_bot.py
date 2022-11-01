# -*- coding: utf-8 -*-
"""
Простой web api для чатбота https://github.com/Koziev/chatbot на Flask.

01.11.2022 Переделка на новую версию диалоговой системы
"""

import sys
import os
import argparse
import logging.handlers
import json

from flask import Flask, request, Response
from flask import jsonify

from ruchatbot.bot.conversation_engine import BotCore, SessionFactory
from ruchatbot.bot.text_utils import TextUtils
from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.bot.bot_profile import BotProfile
from ruchatbot.bot.facts_database import FactsDatabase


flask_app = Flask(__name__)


@flask_app.route('/start_conversation', methods=["GET"])
def start_conversation():
    user = request.args.get('user', 'anonymous')
    session = session_factory.start_conversation(user)
    msg1 = request.args.get('phrase')
    if msg1:
        session.dialog.add_bot_message(msg1)
    else:
        msg1 = bot.start_greeting_scenario(session)

    logging.debug('start_conversation interlocutor="%s" message=〚%s〛', user, msg1)
    #session.add_bot_message(msg1)
    #session.enqueue_replies([msg1])
    return jsonify({'processed': True})


# response = requests.get(chatbot_url + '/' + 'push_phrase?user={}&phrase={}'.format(user_id, phrase))
@flask_app.route('/push_phrase', methods=["GET"])
def service_push():
    user_id = request.args.get('user', 'anonymous')
    phrase = request.args['phrase']
    logging.debug('push_phrase user="%s" phrase=〚%s〛', user_id, phrase)

    session = session_factory.get_session(user_id)
    session.dialog.add_human_message(phrase)

    replies = bot.process_human_message(session)
    #session.enqueue_replies(replies)

    response = {'processed': True}
    return jsonify(response)


@flask_app.route('/pop_phrase', methods=["GET"])
def pop_phrase():
    user = request.args.get('user', 'anonymous')
    session = session_factory.get_session(user)
    reply = session.pop_reply()
    return jsonify({'reply': reply})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REST API for Сhatbot v7')
    parser.add_argument('--chatbot_dir', type=str, default=os.path.expanduser('~/polygon/chatbot'))
    parser.add_argument('--log', type=str, default=os.path.expanduser('~/polygon/chatbot/tmp/conversation_engine.log'))
    parser.add_argument('--profile', type=str, default=os.path.expanduser('~/polygon/chatbot/data/profile_1.json'), help='Path to yaml file with bot persona records')
    parser.add_argument('--bert', type=str)
    parser.add_argument('--db', type=str, default=':memory:', help='Connection string for SQLite storage file; use :memory: for no persistence')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='9098')

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

    # Запуск в режиме rest-сервиса
    listen_ip = args.ip
    listen_port = args.port
    logging.info('Going to run flask_app listening %s:%s profile_path=%s', listen_ip, listen_port, profile_path)
    flask_app.run(debug=True, host=listen_ip, port=listen_port)

# -*- coding: utf-8 -*-
"""
web api для чатбота https://github.com/Koziev/chatbot на Flask.
"""

from __future__ import print_function

import argparse
import logging
import logging.handlers
import os

from bot_service import flask_app
from bot_service.global_params import facts_folder, models_folder,\
    data_folder, w2v_folder

from bot.files3_facts_storage import Files3FactsStorage
from bot.text_utils import TextUtils
from bot.simple_answering_machine import SimpleAnsweringMachine
from bot.bot_scripting import BotScripting


listen_ip = '127.0.0.1'

# настраиваем логирование в файл
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

lf = logging.handlers.RotatingFileHandler('../tmp/bot_service.log',
                                          mode='a',
                                          maxBytes=5000000,
                                          backupCount=5,
                                          encoding='utf-8',
                                          delay=0)
lf.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')
lf.setFormatter(formatter)
logging.getLogger('').addHandler(lf)


@flask_app.before_first_request
def init_chatbot():
    logging.info(u'init_chatbot: models_folder={}'.format(models_folder))

    text_utils = TextUtils()
    text_utils.load_dictionaries(data_folder)

    facts_storage = Files3FactsStorage(text_utils=text_utils,
                                       facts_folder=facts_folder)
    bot = SimpleAnsweringMachine(facts_storage=facts_storage, text_utils=text_utils)
    bot.load_models(models_folder, w2v_folder)

    scripting = BotScripting(data_folder)
    bot.set_scripting(scripting)

    flask_app.config['bot'] = bot
    logging.info('init_chatbot complete')


if __name__ == '__main__':
    # Разбор параметров запуска, указанных в командной строке
    parser = argparse.ArgumentParser(description='Chatbot web-service')
    parser.add_argument('--data_folder', type=str, default='../data')
    parser.add_argument('--w2v_folder', type=str, default='../data')
    parser.add_argument('--facts_folder', type=str, default='../data',
                        help='path to folder containing knowledgebase files')
    parser.add_argument('--models_folder', type=str, default='../tmp', help='path to folder with pretrained models')
    parser.add_argument('--ip', help='listen to specified IP address', type=str, default=listen_ip)

    args = parser.parse_args()
    facts_folder = os.path.expanduser(args.facts_folder)
    models_folder = os.path.expanduser(args.models_folder)
    data_folder = os.path.expanduser(args.data_folder)
    w2v_folder = os.path.expanduser(args.w2v_folder)
    listen_ip = args.ip

    logging.info('Going to run flask_app listening {} models_folder={} facts_folder={} data_folder={} w2v_folder={}'.format(listen_ip, models_folder, facts_folder, data_folder, w2v_folder))
    flask_app.run(debug=True, host=listen_ip)


# https://stackoverflow.com/questions/8495367/using-additional-command-line-arguments-with-gunicorn?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# Gunicorn entry point generator
def start_extractor(*args, **kwargs):
    global facts_folder, models_folder, data_folder, w2v_folder

    # Gunicorn CLI args are useless.
    # https://stackoverflow.com/questions/8495367/
    #
    # Start the application in modified environment.
    # https://stackoverflow.com/questions/18668947/
    #
    for k in kwargs:
        if k == 'facts_folder':
            facts_folder = kwargs[k]
        elif k == 'models_folder':
            models_folder = kwargs[k]
        elif k == 'data_folder':
            data_folder = kwargs[k]
        elif k == 'w2v_folder':
            w2v_folder = kwargs[k]

    return flask_app

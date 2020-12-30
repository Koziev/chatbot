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
from ruchatbot.bot_service.global_params import profile_path, models_folder, data_folder, w2v_folder, chitchat_url
from ruchatbot.frontend.bot_creator import create_chatbot


listen_ip = '127.0.0.1'
listen_port = 9001


@flask_app.before_first_request
def init_chatbot():
    if 'bot' not in flask_app.config:
        logging.info('init_chatbot: profile_path="%s" models_folder="%s" data_folder="%s" chitchat_url="%s"',
                     profile_path, models_folder, data_folder, chitchat_url)

        bot = create_chatbot(profile_path, models_folder, w2v_folder, data_folder, debugging=True, chitchat_url=chitchat_url)

        #def on_order(order_anchor_str, bot, session):
        #    bot.say(session, 'Выполняю команду "{}"'.format(order_anchor_str))
        #bot.on_process_order = on_order

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
    parser.add_argument('--chitchat_url', type=str, help='chit-chat service endpoint')
    parser.add_argument('--debugging', action='store_true', default=False)

    args = parser.parse_args()
    profile_path = os.path.expanduser(args.profile)
    models_folder = os.path.expanduser(args.models_folder)
    data_folder = os.path.expanduser(args.data_folder)
    w2v_folder = os.path.expanduser(args.w2v_folder)
    tmp_folder = os.path.expanduser(args.tmp_folder)
    chitchat_url = args.chitchat_url
    listen_ip = args.ip
    listen_port = args.port

    # настраиваем логирование в файл
    init_trainer_logging(os.path.join(tmp_folder, 'flask_service_bot.log'), debugging=args.debugging)

    if args.preload:
        init_chatbot()

    logging.info('Going to run flask_app listening %s:%d profile_path="%s" models_folder="%s" data_folder="%s" w2v_folder="%s"',
                 listen_ip, listen_port, profile_path, models_folder, data_folder, w2v_folder)
    flask_app.run(debug=args.debugging, host=listen_ip, port=listen_port)


# https://stackoverflow.com/questions/8495367/using-additional-command-line-arguments-with-gunicorn?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
# Gunicorn entry point generator
def start_extractor(*args, **kwargs):
    global profile_path, models_folder, data_folder, w2v_folder, tmp_folder, chitchat_url

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
        elif k == 'chitchat_url':
            chitchat_url = kwargs[k]

    return flask_app

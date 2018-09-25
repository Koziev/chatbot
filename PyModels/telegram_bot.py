# -*- coding: utf-8 -*-
"""
Реализация чатбота для Телеграмма.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability

import logging
import platform
import sys
import argparse
import os

import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

from bot.files3_facts_storage import Files3FactsStorage
from bot.text_utils import TextUtils
from bot.simple_answering_machine import SimpleAnsweringMachine
from bot.console_utils import input_kbd


def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Answering machine is running on '+platform.platform())


def echo(bot, update):
    # update.chat.first_name
    # update.chat.last_name
    try:
        user_id = update.message.chat.first_name+u' '+update.message.chat.last_name
        question = update.message.text

        logging.info(u'Answering to "{}"'.format(question))

        total_answer = u''
        answering_machine.push_phrase(user_id, question)
        while True:
            answer = answering_machine.pop_phrase(user_id)
            if len(answer) == 0:
                break
            else:
                if len(total_answer) > 0:
                    total_answer += u'\n'
                total_answer += answer

        bot.send_message(chat_id=update.message.chat_id, text=total_answer)
        #bot.send_message(chat_id=update.message.chat_id, text=update.message.text)
    except:
        logging.error(sys.exc_info()[0])


# -------------------------------------------------------

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


# Разбор параметров запуска бота, указанных в командной строке

parser = argparse.ArgumentParser(description='Telegram chatbot')
parser.add_argument('--token', type=str, default='', help='Telegram token for bot')
parser.add_argument('--data_folder', type=str, default='../data')
parser.add_argument('--w2v_folder', type=str, default='../data')
parser.add_argument('--facts_folder', type=str, default='../data', help='path to folder containing knowledgebase files')
parser.add_argument('--models_folder', type=str, default='../tmp', help='path to folder with pretrained models')

args = parser.parse_args()

facts_folder = os.path.expanduser(args.facts_folder)
models_folder = os.path.expanduser(args.models_folder)
data_folder = os.path.expanduser(args.data_folder)
w2v_folder = os.path.expanduser(args.w2v_folder)

telegram_token = args.token
if len(telegram_token) == 0:
    telegram_token = input_kbd('Enter Telegram token:')

# -------------------------------------------------------

bot = telegram.Bot(token=telegram_token)
print(bot.getMe())

logging.info('Loading answering machine models...')
text_utils = TextUtils()
text_utils.load_dictionaries(data_folder)

facts_storage = Files3FactsStorage(text_utils=text_utils, facts_folder=facts_folder)

answering_machine = SimpleAnsweringMachine( facts_storage=facts_storage, text_utils=text_utils)
bot.load_models(models_folder, w2v_folder)

# ------------------------------------------------------

updater = Updater(token=telegram_token)
dispatcher = updater.dispatcher

# -------------------------------------------------------

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

echo_handler = MessageHandler(Filters.text, echo)
dispatcher.add_handler(echo_handler)
# -------------------------------------------------------

logging.info('Start polling...')
updater.start_polling()

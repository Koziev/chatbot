# -*- coding: utf-8 -*-
'''
Вопрос-ответная машина подключена к API телеграм, обеспечивая работу чатбота.
'''

from __future__ import print_function
from __future__ import division  # for python2 compatability

import logging
import platform
import sys
import argparse

import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

from files3_facts_storage import Files3FactsStorage
from text_utils import TextUtils
from simple_answering_machine import SimpleAnsweringMachine


TOKEN = '' # токен нужно указать в командной строке запуска бота опцией --token XXX

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
            if len(answer)==0:
                break
            else:
                if len(total_answer)>0:
                    total_answer += u'\n'
                total_answer += answer;

        bot.send_message(chat_id=update.message.chat_id, text=total_answer)
        #bot.send_message(chat_id=update.message.chat_id, text=update.message.text)
    except:
        logging.error(sys.exc_info()[0])


# -------------------------------------------------------

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


# Разбор параметров запуска бота, указанных в командной строке

parser = argparse.ArgumentParser(description='Run Telegram bot')
parser.add_argument('--token', help='Telegram token for bot')
args = parser.parse_args()

TOKEN = args.token

# -------------------------------------------------------

bot = telegram.Bot(token=TOKEN)

print(bot.getMe())

# ------------------------------------------------------

logging.info('Loading answering machine models...')
text_utils = TextUtils()
facts_storage = Files3FactsStorage(text_utils=text_utils, facts_folder='/home/eek/polygon/paraphrasing/data')
answering_machine = SimpleAnsweringMachine( facts_storage=facts_storage, text_utils=text_utils)
answering_machine.load_models('/home/eek/polygon/paraphrasing/tmp')
# bot.answering_machine = answering_machine


# ------------------------------------------------------

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

# -------------------------------------------------------

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

echo_handler = MessageHandler(Filters.text, echo)
dispatcher.add_handler(echo_handler)
# -------------------------------------------------------

logging.info('Start polling...')
updater.start_polling()

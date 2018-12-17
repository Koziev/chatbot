# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для диалогового движка (https://github.com/Koziev/chatbot).
"""

import os
import argparse

from bot.files3_facts_storage import Files3FactsStorage
from bot.text_utils import TextUtils
from bot.simple_answering_machine import SimpleAnsweringMachine
from bot.console_utils import input_kbd, print_answer, print_tech_banner
from bot.bot_scripting import BotScripting
from bot.bot_personality import BotPersonality
from bot.order_comprehension_table import OrderComprehensionTable
from utils.logging_helpers import init_trainer_logging


user_id = 'test'

parser = argparse.ArgumentParser(description='Question answering machine')
parser.add_argument('--data_folder', type=str, default='../data')
parser.add_argument('--w2v_folder', type=str, default='../data')
parser.add_argument('--facts_folder', type=str, default='../data', help='path to folder containing knowledgebase files')
parser.add_argument('--models_folder', type=str, default='../tmp', help='path to folder with pretrained models')
parser.add_argument('--tmp_folder', type=str, default='../tmp', help='path to folder for logfile etc')

args = parser.parse_args()
facts_folder = os.path.expanduser(args.facts_folder)
models_folder = os.path.expanduser(args.models_folder)
data_folder = os.path.expanduser(args.data_folder)
w2v_folder = os.path.expanduser(args.w2v_folder)
tmp_folder = os.path.expanduser(args.tmp_folder)

init_trainer_logging(os.path.join(tmp_folder, 'console_chatbot.log'))

# Создаем необходимое окружение для бота
# Инструменты для работы с текстом, включая морфологию и таблицы словоформ.
text_utils = TextUtils()
text_utils.load_dictionaries(data_folder)

# Инициализируем движок вопросно-ответной системы. Он может обслуживать несколько
# ботов, хотя тут у нас будет работать только один.
machine = SimpleAnsweringMachine(text_utils=text_utils)
machine.load_models(models_folder, w2v_folder)
machine.trace_enabled = True  # для отладки

# Конкретная реализация хранилища фактов.
facts_storage = Files3FactsStorage(text_utils=text_utils,
                                   facts_folder=facts_folder)

scripting = BotScripting(data_folder)

# Инициализируем бота
bot = BotPersonality(bot_id='test_bot',
                     engine=machine,
                     facts=facts_storage,
                     scripting=scripting,
                     enable_scripting=False,
                     enable_smalltalk=False)

oct = OrderComprehensionTable()
oct.load_file(os.path.join(data_folder, 'orders.txt'))  # загружаем таблицу интерпретации приказов
bot.set_order_templates(oct)


def on_order(order_anchor_str, bot, session):
    bot.say(session, u'Выполняю команду \"{}\"'.format(order_anchor_str))


bot.on_process_order = on_order

print_tech_banner()

bot.start_conversation(user_id)

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

    question = input_kbd('H:>').lower()
    if len(question) == 0:
        break
    bot.push_phrase(user_id, question)

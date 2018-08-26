# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для диалогового движка (https://github.com/Koziev/chatbot).
"""

import logging
import argparse

from bot.files3_facts_storage import Files3FactsStorage
from bot.text_utils import TextUtils
from bot.simple_answering_machine import SimpleAnsweringMachine
from bot.console_utils import input_kbd, print_answer, print_tech_banner
from bot.bot_scripting import BotScripting


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.ERROR)

user_id = 'test'

parser = argparse.ArgumentParser(description='Question answering machine')
parser.add_argument('--data_folder', type=str, default='../../data')
parser.add_argument('--w2v_folder', type=str, default='../../data')
parser.add_argument('--facts_folder', type=str, default='../../data', help='path to folder containing knowledgebase files')
parser.add_argument('--models_folder', type=str, default='../../tmp', help='path to folder with pretrained models')

args = parser.parse_args()
facts_folder = args.facts_folder
models_folder = args.models_folder
data_folder = args.data_folder
w2v_folder = args.w2v_folder

text_utils = TextUtils()
text_utils.load_dictionaries(data_folder)

facts_storage = Files3FactsStorage(text_utils=text_utils,
                                   facts_folder=facts_folder)
bot = SimpleAnsweringMachine(facts_storage=facts_storage, text_utils=text_utils)
bot.load_models(models_folder, w2v_folder)

scripting = BotScripting(data_folder)
bot.set_scripting(scripting)

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

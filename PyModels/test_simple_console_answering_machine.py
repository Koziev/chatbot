# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для диалогового движка (https://github.com/Koziev/chatbot).
"""

import colorama  # https://pypi.python.org/pypi/colorama
import platform
import sys
import os
import logging
import argparse

from bot.files3_facts_storage import Files3FactsStorage
from bot.text_utils import TextUtils
from bot.simple_answering_machine import SimpleAnsweringMachine


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

user_id = 'test'


parser = argparse.ArgumentParser(description='Question answering machine')
parser.add_argument('--facts_folder', type=str, default='../../data', help='path to folder containing knowledgebase files')
parser.add_argument('--models_folder', type=str, default='../../tmp', help='path to folder with pretrained models')

args = parser.parse_args()
facts_folder = args.facts_folder
models_folder = args.models_folder

text_utils = TextUtils()
facts_storage = Files3FactsStorage(text_utils=text_utils, facts_folder=facts_folder)
bot = SimpleAnsweringMachine(facts_storage=facts_storage, text_utils=text_utils)
bot.load_models(models_folder)

print( colorama.Fore.LIGHTBLUE_EX+'Answering machine is running on '+platform.platform()+colorama.Fore.RESET )

while True:
    print('\n')
    question = raw_input('Q:> ').decode(sys.stdout.encoding).strip().lower()
    if len(question) == 0:
        break
    bot.push_phrase(user_id, question)
    while True:
        answer = bot.pop_phrase(user_id)
        if len(answer) == 0:
            break

        print(u'A:> '+colorama.Fore.GREEN + u'{}'.format(answer)+colorama.Fore.RESET)

# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для диалогового движка (https://github.com/Koziev/chatbot).
"""

from files3_facts_storage import Files3FactsStorage
from text_utils import TextUtils
from simple_answering_machine import SimpleAnsweringMachine
import colorama  # https://pypi.python.org/pypi/colorama
import platform
import sys


user_id = 'test'

text_utils = TextUtils()
facts_storage = Files3FactsStorage(text_utils=text_utils, facts_folder='/home/eek/polygon/paraphrasing/data')
bot = SimpleAnsweringMachine( facts_storage=facts_storage, text_utils=text_utils)
bot.load_models('/home/eek/polygon/paraphrasing/tmp')

print( colorama.Fore.LIGHTBLUE_EX+'Answering machine is running on '+platform.platform()+colorama.Fore.RESET )

while True:
    print('\n')
    question = raw_input('Q:> ').decode(sys.stdout.encoding).strip().lower()
    if len(question)==0:
        break
    bot.push_phrase(user_id, question)
    while True:
        answer = bot.pop_phrase(user_id)
        if len(answer)==0:
            break

        print(u'A:> '+colorama.Fore.GREEN + u'{}'.format(answer)+colorama.Fore.RESET)

# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для простого "да-нет" бота без хранилища фактов.
Для ручного тестирования моделей.
Часть проекта вопросно-ответной системы https://github.com/Koziev/chatbot 
"""

import logging
import argparse

from bot.text_utils import TextUtils
from bot.console_utils import input_kbd, print_answer, print_error
from bot.yes_no_bot import YesNoBot

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.ERROR)

parser = argparse.ArgumentParser(description='yes/no answering machine')
parser.add_argument('--models_folder', type=str, default='../tmp', help='path to folder with pretrained models')

args = parser.parse_args()
models_folder = args.models_folder

# В консоли пользователи вводит пары фраз - предпосылку и вопрос, на который можно ответить на
# основании факты, выражаемого предпосылкой. Движок бота сначала оценивает релевантность
# введенной предпосылки и вопроса. Если релевантность слишком низка - выводится ответ "не определено".
# Иначе определяется вариант ответа - "да" или "нет".

text_utils = TextUtils()
bot = YesNoBot(text_utils=text_utils)
bot.load_models(models_folder)

max_nb_premises = 1

while True:
    print('\n')
    phrases = []
    while True:
        phrase = input_kbd(u':> ')
        if len(phrase) > 0:
            phrases.append(phrase)
            if phrase[-1] == u'?':
                break
            elif len(phrases) == max_nb_premises+1:
                break
        elif len(phrases) > 1:
            break

    if len(phrases) < 2:
        print_error('At least 2 phrases expected!')
        continue

    premises = phrases[:-1]
    question = phrases[-1]

    answer = bot.infer_answer(premises, question)
    print_answer(u'A:>', answer)

# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для простого "да-нет" бота без хранилища фактов.
Для ручного тестирования моделей.
Часть проекта вопросно-ответной системы https://github.com/Koziev/chatbot 
"""

import logging
import argparse
import os

from bot.text_utils import TextUtils
from bot.console_utils import input_kbd, print_answer, print_error
from bot.yes_no_bot import YesNoBot

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.ERROR)

parser = argparse.ArgumentParser(description='yes/no answering machine')
parser.add_argument('--models_folder', type=str, default='../tmp', help='path to folder with pretrained models')
parser.add_argument('--w2v_folder', type=str, default=os.path.expanduser('~/polygon/w2v'))

args = parser.parse_args()
models_folder = args.models_folder
w2v_folder = args.w2v_folder

# В консоли пользователи вводит фразы - предпосылку(и) и вопрос, на который можно ответить на
# основании фактов, выражаемых предпосылкой. Движок бота сначала оценивает релевантность
# введенной предпосылки и вопроса. Если релевантность слишком низка - выводится ответ "не определено".
# Иначе определяется вариант ответа - "да" или "нет".

text_utils = TextUtils()
bot = YesNoBot(text_utils=text_utils)
bot.load_models(models_folder, w2v_folder)

max_nb_premises = 4

while True:
    print('\n')
    phrases = []
    while True:
        phrase = input_kbd(u'H:>')
        if len(phrase) > 0:
            phrases.append(phrase)
            if phrase[-1] == u'?':
                break
            elif len(phrases) == max_nb_premises+1:
                break
        elif len(phrases) > 1:
            break

    premises = phrases[:-1]  # от 0 до нескольких предпосылок
    question = phrases[-1]  # последнее введенное - вопрос

    answer = bot.infer_answer(premises, question)
    print_answer(u'B:>', answer)

# -*- coding: utf-8 -*-
"""
Простейший консольный фронтэнд для FAQ-бота (https://github.com/Koziev/chatbot).
"""

import os
import argparse

from bot.text_utils import TextUtils
from bot.console_utils import input_kbd
from bot.simple_faq_bot import Simple_FAQ_Bot
from utils.logging_helpers import init_trainer_logging

parser = argparse.ArgumentParser(description='FAQ bot')
parser.add_argument('--data_folder', type=str, default='../data')
parser.add_argument('--w2v_folder', type=str, default='~/polygon/w2v')
parser.add_argument('--models_folder', type=str, default='../tmp', help='path to folder with pretrained models')
parser.add_argument('--tmp_folder', type=str, default='../tmp', help='path to folder for logfile etc')

args = parser.parse_args()
models_folder = os.path.expanduser(args.models_folder)
data_folder = os.path.expanduser(args.data_folder)
w2v_folder = os.path.expanduser(args.w2v_folder)
tmp_folder = os.path.expanduser(args.tmp_folder)

init_trainer_logging(os.path.join(tmp_folder, 'console_faq_bot.log'))

# Создаем необходимое окружение для бота
# Инструменты для работы с текстом, включая морфологию и таблицы словоформ.
text_utils = TextUtils()
#text_utils.load_dictionaries(data_folder)

machine = Simple_FAQ_Bot(text_utils)
machine.load_models(models_folder, w2v_folder)

# Загружаем из файла условные пары вопрос-ответ.
machine.load_faq(os.path.join(data_folder, 'faq2.txt'))

while True:
    print('\n')
    question = input_kbd('H:>').lower()
    if len(question) == 0:
        break
    faq_answer, confidence, faq_question = machine.select_answer(question)
    if confidence > 0.5:
        print(u'{} {}'.format(confidence, faq_answer))

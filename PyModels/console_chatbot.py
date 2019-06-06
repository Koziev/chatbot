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
from bot.plain_file_faq_storage import PlainFileFaqStorage
from utils.logging_helpers import init_trainer_logging


user_id = 'test'

parser = argparse.ArgumentParser(description='Question answering machine')
parser.add_argument('--data_folder', type=str, default='../data')
parser.add_argument('--w2v_folder', type=str, default='../tmp')
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


# Создаем необходимое окружение для бота.
# Инструменты для работы с текстом, включая морфологию и таблицы словоформ.
text_utils = TextUtils()
text_utils.load_dictionaries(data_folder, models_folder)

scripting = BotScripting(data_folder)
scripting.load_rules(os.path.join(data_folder, 'rules.yaml'),
                     os.path.join(models_folder, 'smalltalk_generative_grammars.bin'),
                     text_utils
                     )



# Инициализируем движок вопросно-ответной системы. Он может обслуживать несколько
# ботов с разными базами фактов и правил, хотя тут у нас будет работать только один.
machine = SimpleAnsweringMachine(text_utils=text_utils)
machine.load_models(data_folder, models_folder, w2v_folder)
machine.trace_enabled = True  # для отладки

# Конкретная реализация хранилища фактов - плоские файлы с utf-8, без форматирования.
facts_storage = Files3FactsStorage(text_utils=text_utils,
                                   facts_folder=facts_folder)

# Подключем простое файловое хранилище с FAQ-правилами бота.
# Движок бота сопоставляет вопрос пользователя с опорными вопросами в FAQ базе,
# и если нашел хорошее соответствие (синонимичность выше порога), то
# выдает ответную часть найденной записи.
faq = PlainFileFaqStorage(os.path.join(data_folder, 'faq2.txt'))

# Инициализируем бота, загружаем правила (файл data/rules.yaml).
bot = BotPersonality(bot_id='test_bot',
                     engine=machine,
                     facts=facts_storage,
                     faq=faq,
                     scripting=scripting,
                     enable_scripting=True,
                     enable_smalltalk=True)


def on_order(order_anchor_str, bot, session):
    bot.say(session, u'Выполняю команду \"{}\"'.format(order_anchor_str))
    # Всегда возвращаем True, как будто можем выполнить любой приказ.
    # В реальных сценариях нужно вернуть False, если приказ не опознан
    return True


bot.on_process_order = on_order


def on_weather_forecast(bot, session, user_id, interpreted_phrase):
    when_arg = bot.extract_entity(u'когда', interpreted_phrase)
    return u'Прогноз погоды на момент времени "{}" сгенерирован в функции on_weather_forecast для демонстрации'.format(when_arg)


bot.add_event_handler(u'weather_forecast', on_weather_forecast)


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

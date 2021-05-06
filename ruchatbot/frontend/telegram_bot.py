# -*- coding: utf-8 -*-
"""
Реализация чатбота для Телеграмма.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

import logging
import argparse
import os

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update

from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.frontend.bot_creator import create_chatbot, ChitchatConfig


def get_interlocutor_id(update: Update) -> str:
    # В качестве идентификатора сессии собеседника можно брать id чата (update.message.chat_id)
    # или id пользователя (update.from_user.id). Чтобы бот узнавал пользователя в разных чатах,
    # будем использовать id пользователя.
    # user_id = str(update.message.chat_id)
    user_id = str(update.message.from_user.id)
    return user_id


def start(update, context) -> None:
    user_id = get_interlocutor_id(update)
    logging.debug('Entering START callback with user_id=%s', user_id)
    chatbot.start_conversation(user_id)

    while True:
        answer = chatbot.pop_phrase(user_id)
        if len(answer) == 0:
            break

        context.bot.send_message(chat_id=update.message.chat_id, text=answer)
    logging.debug('Leaving START callback with user_id=%s', user_id)


def echo(update, context):
    # update.chat.first_name
    # update.chat.last_name
    try:
        # В качестве идентификатора сессии собеседника можно брать id чата (update.message.chat_id)
        # или id пользователя (update.from_user.id). Чтобы бот узнавал пользователя в разных чатах,
        # будем использовать id пользователя.
        #user_id = str(update.message.chat_id)
        user_id = get_interlocutor_id(update)

        question = update.message.text

        logging.info('Answering to "%s" for user=%s id="%s" in chat=%s', question, update.message.from_user.name, user_id, str(update.message.chat_id))

        chatbot.push_phrase(user_id, question)
        while True:
            answer = chatbot.pop_phrase(user_id)
            if len(answer) == 0:
                break
            else:
                context.bot.send_message(chat_id=update.message.chat_id, text=answer)
    except Exception as ex:
        logging.error(ex)  # sys.exc_info()[0]


if __name__ == '__main__':
    # Разбор параметров запуска бота, указанных в командной строке
    parser = argparse.ArgumentParser(description='Telegram chatbot')
    parser.add_argument('--token', type=str, default='', help='Telegram token for bot')
    parser.add_argument('--profile', type=str, help='path to profile file')
    parser.add_argument('--data_folder', type=str, default='../../data')
    parser.add_argument('--w2v_folder', type=str, default='../../tmp')
    parser.add_argument('--models_folder', type=str, default='../../tmp', help='path to folder with pretrained models')
    parser.add_argument('--tmp_folder', type=str, default='../../tmp', help='path to folder for logfile etc')
    parser.add_argument('--chitchat_url', type=str, help='chit-chat service endpoint')

    args = parser.parse_args()

    profile_path = os.path.expanduser(args.profile) if args.profile is not None else None
    models_folder = os.path.expanduser(args.models_folder)
    data_folder = os.path.expanduser(args.data_folder)
    w2v_folder = os.path.expanduser(args.w2v_folder)
    tmp_folder = os.path.expanduser(args.tmp_folder)

    #logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    init_trainer_logging(os.path.join(tmp_folder, 'telegram_bot.log'), True)

    # Для работы с сервером телеграмма нужен зарегистрированный бот.
    # Результатом регистрации является токен - уникальная строка символов.
    # Вот эту строку надо сейчас ввести с консоли.
    # Возможно, следует предусмотреть передачу токена через опцию ком. строки.
    telegram_token = args.token
    if len(telegram_token) == 0:
        telegram_token = input('Enter Telegram token:> ').strip()

    # Задать используемый профиль можно с консоли.
    while not profile_path:
        profile = input('Choose profile [1, 2, 3]:> ').strip()
        if profile == '1':
            profile_path = os.path.join(data_folder, 'profile_1.json')
        elif profile == '2':
            profile_path = os.path.join(data_folder, 'profile_2.json')
        elif profile == '3':
            profile_path = os.path.join(data_folder, 'profile_3.json')

    tg_bot = telegram.Bot(token=telegram_token).getMe()
    bot_id = tg_bot.name
    logging.info('Telegram bot "%s" id=%s', tg_bot.name, tg_bot.id)

    if args.chitchat_url:
        rugpt_chitchat_config = ChitchatConfig()
        rugpt_chitchat_config.service_endpoint = args.chitchat_url
        # rugpt_chitchat_config.temperature = 0.9
        rugpt_chitchat_config.num_return_sequences = 2
    else:
        rugpt_chitchat_config = None

    logging.debug('Bot loading: profile="%s"', profile_path)
    chatbot = create_chatbot(profile_path, models_folder, w2v_folder, data_folder, True, bot_id=bot_id,
                             chitchat_config=rugpt_chitchat_config)

    updater = Updater(token=telegram_token)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    echo_handler = MessageHandler(Filters.text, echo)
    dispatcher.add_handler(echo_handler)

    logging.getLogger('telegram.bot').setLevel(logging.INFO)
    logging.getLogger('telegram.vendor.ptb_urllib3.urllib3.connectionpool').setLevel(logging.INFO)

    logging.info('Start polling messages for bot %s', tg_bot.name)
    updater.start_polling()
    updater.idle()

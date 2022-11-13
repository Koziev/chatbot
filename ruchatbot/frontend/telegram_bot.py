"""
Реализация чатбота для Телеграмма.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.

01.11.2022 переделка для новой версии диалоговой системы
13.11.2022 втаскиваем скриптование
"""

import os
import argparse
import logging.handlers
import json
import traceback
import getpass

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

from ruchatbot.bot.conversation_engine import BotCore, SessionFactory
from ruchatbot.bot.text_utils import TextUtils
from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.bot.bot_profile import BotProfile
from ruchatbot.bot.facts_database import FactsDatabase
from ruchatbot.scripting.bot_scripting import BotScripting


def get_user_id(update: Update) -> str:
    user_id = str(update.message.from_user.id)
    return user_id


def tg_start(update, context) -> None:
    user_id = get_user_id(update)
    logging.debug('Entering START callback with user_id=%s', user_id)

    session = session_factory.start_conversation(user_id)

    msg1 = bot.start_greeting_scenario(session)
    context.bot.send_message(chat_id=update.message.chat_id, text=msg1)
    logging.debug('Leaving START callback with user_id=%s', user_id)


LIKE = '_Like_'
DISLIKE = '_Dislike_'

last_bot_reply = dict()


def tg_echo(update, context):
    # update.chat.first_name
    # update.chat.last_name
    try:
        user_id = get_user_id(update)

        session = session_factory.get_session(user_id)

        if update.message.text == LIKE:
            logging.info('LIKE user_id="%s" bot_reply=〚%s〛', user_id, last_bot_reply[user_id])
            return

        if update.message.text == DISLIKE:
            logging.info('DISLIKE user_id="%s" bot_reply=〚%s〛', user_id, last_bot_reply[user_id])
            return

        q = update.message.text
        logging.info('Will reply to 〚%s〛 for user="%s" id=%s in chat=%s', q, update.message.from_user.name, user_id, str(update.message.chat_id))

        session.dialog.add_human_message(q)
        replies = bot.process_human_message(session)
        for reply in replies:
            logging.info('Bot reply=〚%s〛 to user="%s"', reply, user_id)

        keyboard = [[LIKE, DISLIKE]]
        reply_markup = ReplyKeyboardMarkup(keyboard,
                                           one_time_keyboard=True,
                                           resize_keyboard=True,
                                           per_user=True)

        context.bot.send_message(chat_id=update.message.chat_id, text=reply, reply_markup=reply_markup)
        last_bot_reply[user_id] = reply
    except Exception as ex:
        logging.error(ex)  # sys.exc_info()[0]
        logging.error('Error occured when message 〚%s〛 from interlocutor "%s" was being processed: %s', update.message.text, user_id, traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Telegram Bot for Сhatbot v7')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--chatbot_dir', type=str, default=os.path.expanduser('~/polygon/chatbot'))
    parser.add_argument('--log', type=str, default=os.path.expanduser('~/polygon/chatbot/tmp/conversation_engine.log'))
    parser.add_argument('--profile', type=str, default=os.path.expanduser('~/polygon/chatbot/data/profile_1.json'), help='Path to yaml file with bot persona records')
    parser.add_argument('--bert', type=str)
    parser.add_argument('--db', type=str, default=':memory:', help='Connection string for SQLite storage file; use :memory: for no persistence')

    args = parser.parse_args()

    chatbot_dir = args.chatbot_dir
    models_dir = os.path.join(chatbot_dir, 'tmp')
    data_dir = os.path.join(chatbot_dir, 'data')
    tmp_dir = os.path.join(chatbot_dir, 'tmp')
    profile_path = args.profile

    init_trainer_logging(args.log, True)

    # Настроечные параметры бота собраны в профиле - файле в json формате.
    bot_profile = BotProfile("bot_v7")
    bot_profile.load(profile_path, data_dir, models_dir)

    text_utils = TextUtils()
    text_utils.load_dictionaries(data_dir, models_dir)

    scripting = BotScripting()
    scripting.load_resources(bot_profile, text_utils)
    bot_profile.scripting = scripting

    # 19-03-2022 запрещаем тензорфлоу резервировать всю память в гпу по дефолту, так как
    # это мешает потом нормально работать моделям на торче.
    #for gpu in tf.config.experimental.list_physical_devices('GPU'):
    #    tf.config.experimental.set_memory_growth(gpu, True)

    bot = BotCore()
    if args.bert is not None:
        bot.load_bert(args.bert)
    else:
        # Определяем модель берта по конфигу детектора полных предпосылок.
        with open(os.path.join(models_dir, 'closure_detector_2.cfg'), 'r') as f:
            cfg = json.load(f)
            bert_name = cfg['bert_model']
            bot.load_bert(bert_name)

    bot.load(models_dir, text_utils)

    # Хранилище новых фактов, извлекаемых из диалоговых сессий.
    # По умолчанию размещается только в оперативной памяти.
    # Если в командной строке задана опция --db XXX, то будет использоваться БД sqlite для хранения новых фактов.
    facts_db = FactsDatabase(args.db)

    # Фабрика для создания новых диалоговых сессий и хранения текущих сессий для всех онлайн-собеседников
    session_factory = SessionFactory(bot_profile, text_utils, facts_db)

    logging.info('Starting telegram bot')

    telegram_token = args.token
    if len(telegram_token) == 0:
        telegram_token = getpass.getpass('Enter Telegram token:> ').strip()

    # Телеграм-версия генератора
    tg_bot = telegram.Bot(token=telegram_token).getMe()
    bot_id = tg_bot.name
    logging.info('Telegram bot "%s" id=%s', tg_bot.name, tg_bot.id)

    updater = Updater(token=telegram_token)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', tg_start)
    dispatcher.add_handler(start_handler)

    echo_handler = MessageHandler(Filters.text, tg_echo)
    dispatcher.add_handler(echo_handler)

    logging.getLogger('telegram.bot').setLevel(logging.INFO)
    logging.getLogger('telegram.vendor.ptb_urllib3.urllib3.connectionpool').setLevel(logging.INFO)

    logging.info('Start polling messages for bot %s', tg_bot.name)
    updater.start_polling()
    updater.idle()

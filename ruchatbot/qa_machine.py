# -*- coding: utf-8 -*-
"""
Создание экземпляра stateless-бота, работающего исключительно в режиме
retrieval-based question answering.
Для проекта https://github.com/Koziev/chatbot
"""

import logging

from .bot.bot_profile import BotProfile
from .bot.profile_facts_reader import ProfileFactsReader
from .bot.text_utils import TextUtils
from .bot.simple_answering_machine import SimpleAnsweringMachine
from .bot.bot_scripting import BotScripting
from .bot.bot_personality import BotPersonality
from .bot.plain_file_faq_storage import PlainFileFaqStorage


def create_qa_bot(profile_path, models_folder, data_folder, w2v_folder,
                  bot_id='test_bot', debugging=False):
    logging.debug('Loading chatbot dictionaries...')
    # Инструменты для работы с текстом, включая морфологию и таблицы словоформ.
    text_utils = TextUtils()
    text_utils.load_dictionaries(data_folder, models_folder)

    # Настроечные параметры аватара собраны в профиле - файле в json формате.
    profile = BotProfile()
    profile.load(profile_path, data_folder, models_folder)

    # Инициализируем движок вопросно-ответной системы. Он может обслуживать несколько
    # ботов с разными провилями (базами фактов и правил), хотя тут у нас будет работать только один.
    machine = SimpleAnsweringMachine(text_utils=text_utils)
    machine.load_models(data_folder, models_folder, w2v_folder)
    machine.trace_enabled = debugging

    # Контейнер для правил
    scripting = BotScripting(data_folder)
    scripting.load_rules(profile.rules_path, profile.smalltalk_generative_rules, text_utils)

    # Конкретная реализация хранилища фактов - плоские файлы в utf-8, с минимальным форматированием
    profile_facts = ProfileFactsReader(text_utils=text_utils, profile_path=profile.premises_path)

    # Подключем простое файловое хранилище с FAQ-правилами бота.
    # Движок бота сопоставляет вопрос пользователя с опорными вопросами в FAQ базе,
    # и если нашел хорошее соответствие (синонимичность выше порога), то
    # выдает ответную часть найденной записи.
    faq_storage = PlainFileFaqStorage(profile.faq_path)

    # Инициализируем аватара
    bot = BotPersonality(bot_id=bot_id,
                         engine=machine,
                         facts=profile_facts,
                         faq=faq_storage,
                         scripting=scripting,
                         enable_scripting=profile.rules_enabled,
                         enable_smalltalk=False,
                         force_question_answering=True)

    logging.debug('Bot instance initialized')
    return bot

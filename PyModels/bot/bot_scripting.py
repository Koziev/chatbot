# -*- coding: utf-8 -*-

import codecs
import os
import random
import logging

from interpreted_phrase import InterpretedPhrase

class BotScripting:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def buid_answer(self, answering_machine, interlocutor, interpreted_phrase):
        return answering_machine.text_utils.language_resources[u'не знаю']

    def generate_response4nonquestion(self, answering_machine, interlocutor, interpreted_phrase):
        '''Генерация реплики для не-вопроса собеседника'''
        return None

    def start_conversation(self, chatbot, session):
        # Начало общения с пользователем, для которого загружена сессия session
        # со всей необходимой инофрмацией - история прежних бесед и т.д
        # Выберем одну из типовых фраз в файле smalltalk_opening.txt
        logging.info(u'BotScripting::start_conversation')
        phrases = []
        filepath = os.path.join(self.data_folder, 'smalltalk_opening.txt')
        if os.path.exists(filepath):
            logging.info(u'Loading greetings from {}'.format(filepath))
            with codecs.open(filepath, 'r', 'utf-8') as rdr:
                for line in rdr:
                    phrase = line.strip()
                    if len(phrase) > 0:
                        phrases.append(phrase)
        else:
            logging.error(u'Greetings file {} does not exist'.format(filepath))

        if len(phrases) > 0:
            return random.choice(phrases)

        return None

    def generate_after_answer(self, answering_machine, interlocutor, interpreted_phrase, answer):
        # todo: потом вынести реализацию в производный класс, чтобы тут осталась только
        # пустая заглушка метода.

        language_resources = answering_machine.text_utils.language_resources
        probe_query_str = language_resources[u'как тебя зовут']
        probe_query = InterpretedPhrase(probe_query_str)
        answer, answer_confidense = answering_machine.build_answer0(interlocutor, probe_query)
        if answer_confidense < 0.70:
            # имя собеседника неизвестно.
            return language_resources[u'А как тебя зовут?']

        return None
# -*- coding: utf-8 -*-

import random
import logging
#import parser
import yaml
import io

from bot.interpreted_phrase import InterpretedPhrase
from smalltalk_replicas import SmalltalkReplicas
from comprehension_table import ComprehensionTable


# TODO вынести в отдельный файл
class ScriptingRule(object):
    def __init__(self, condition, action):
        self.condition = condition
        self.action = action

    def check_condition(self, interpreted_phrase):
        if u'intent' in self.condition:
            return self.condition[u'intent'] == interpreted_phrase.intent
        else:
            raise NotImplementedError()

    def do_action(self, bot, session, user_id, interpreted_phrase):
        if u'say' in self.action:
            if isinstance(self.action[u'say'], list):
                bot.say(session, random.choice(self.action[u'say']))
            else:
                bot.say(session, self.action[u'say'])
        elif u'answer' in self.action:
            if isinstance(self.action[u'answer'], list):
                bot.push_phrase(user_id, random.choice(self.action[u'answer']))
            else:
                bot.push_phrase(user_id, self.action[u'answer'])
        elif u'callback' in self.action:
            resp = bot.invoke_callback(self.action[u'callback'], session, user_id, interpreted_phrase)
            if resp:
                bot.say(session, resp)
        else:
            raise NotImplementedError()


class BotScripting(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.rules = []
        self.greetings = []
        self.goodbyes = []
        self.smalltalk_rules = []
        self.comprehension_rules = None

    @staticmethod
    def __get_node_list(node):
        if isinstance(node, list):
            return node
        else:
            return [node]

    def load_rules(self, yaml_path):
        with io.open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if 'greeting' in data:
                self.greetings = data['greeting']

            if 'goodbye' in data:
                self.goodbyes = data['goodbye']

            for rule in data['rules']:
                # Пока делаем самый простой формат правил - с одним условием и одним актором.
                condition = rule['rule']['if']
                action = rule['rule']['then']

                # Простые правила, которые задают срабатывание по тексту фразы, добавляем в отдельный
                # список, чтобы обрабатывать в модели синонимичности одним пакетом.
                if 'text' in condition and 'say' in action:
                    for condition1 in BotScripting.__get_node_list(condition['text']):
                        rule11 = SmalltalkReplicas(condition1)
                        for answer1 in BotScripting.__get_node_list(action['say']):
                            rule11.add_answer(answer1)
                        self.smalltalk_rules.append(rule11)
                else:
                    rule = ScriptingRule(condition, action)
                    self.rules.append(rule)

            self.comprehension_rules = ComprehensionTable()
            self.comprehension_rules.load_yaml_data(data)


    def enumerate_smalltalk_replicas(self):
        return self.smalltalk_rules

    def buid_answer(self, answering_machine, interlocutor, interpreted_phrase):
        return answering_machine.text_utils.language_resources[u'не знаю']

    def generate_response4nonquestion(self, answering_machine, interlocutor, interpreted_phrase):
        '''Генерация реплики для не-вопроса собеседника'''
        return None

    def start_conversation(self, chatbot, session):
        # Начало общения с пользователем, для которого загружена сессия session
        # со всей необходимой информацией - история прежних бесед и т.д
        # Выберем одну из типовых фраз в файле smalltalk_opening.txt
        logging.info(u'BotScripting::start_conversation')
        if len(self.greetings) > 0:
            return random.choice(self.greetings)

        return None

    def generate_after_answer(self, bot, answering_machine, interlocutor, interpreted_phrase, answer):
        # todo: потом вынести реализацию в производный класс, чтобы тут осталась только
        # пустая заглушка метода.

        language_resources = answering_machine.text_utils.language_resources
        probe_query_str = language_resources[u'как тебя зовут']
        probe_query = InterpretedPhrase(probe_query_str)
        answers, answer_confidenses = answering_machine.build_answers0(bot, interlocutor, probe_query)
        ask_name = False
        if len(answers) > 0:
            if answer_confidenses[0] < 0.70:
                ask_name = True
        else:
            ask_name = True

        if ask_name:
            # имя собеседника неизвестно.
            q = language_resources[u'А как тебя зовут?']
            nq = answering_machine.get_session(bot, interlocutor).count_bot_phrase(q)
            if nq < 3:  # Не будем спрашивать более 2х раз.
                return q

        return None

    def apply_rule(self, bot, session, user_id, interpreted_phrase):
        for rule in self.rules:
            if rule.check_condition(interpreted_phrase):
                rule.do_action(bot, session, user_id, interpreted_phrase)
                return True
        return False

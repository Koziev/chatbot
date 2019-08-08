# -*- coding: utf-8 -*-

import random
import logging
#import parser
import yaml
import io
import pickle

from ruchatbot.bot.interpreted_phrase import InterpretedPhrase
from ruchatbot.bot.smalltalk_rules import SmalltalkSayingRule
from ruchatbot.bot.smalltalk_rules import SmalltalkGeneratorRule
from ruchatbot.bot.smalltalk_rules import SmalltalkRules
from ruchatbot.generative_grammar.generative_grammar_engine import GenerativeGrammarEngine
from ruchatbot.bot.comprehension_table import ComprehensionTable
from ruchatbot.bot.scripting_rule import ScriptingRule
from ruchatbot.bot.verbal_form import VerbalForm
from ruchatbot.bot.scenario import Scenario


class BotScripting(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.greetings = []
        self.goodbyes = []
        self.insteadof_rules = []
        self.smalltalk_rules = []
        self.comprehension_rules = None
        self.forms = []  # список экземпляров VerbalForm
        self.scenarios = []  # список экземпляров Scenario

    @staticmethod
    def __get_node_list(node):
        if isinstance(node, list):
            return node
        else:
            return [node]

    def load_rules(self, yaml_path, compiled_grammars_path, text_utils):
        with io.open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if 'greeting' in data:
                self.greetings = data['greeting']

            if 'goodbye' in data:
                self.goodbyes = data['goodbye']

            if 'forms' in data:
                for form_node in data['forms']:
                    form = VerbalForm.from_yaml(form_node['form'])
                    self.forms.append(form)

            # Для smalltalk-правил нужны скомпилированные генеративные грамматики.
            smalltalk_rule2grammar = dict()
            with open(compiled_grammars_path, 'rb') as f:
                n_rules = pickle.load(f)
                for _ in range(n_rules):
                    key = pickle.load(f)
                    grammar = GenerativeGrammarEngine.unpickle_from(f)
                    grammar.set_dictionaries(text_utils.gg_dictionaries)
                    smalltalk_rule2grammar[key] = grammar

            if 'scenarios' in data:
                for scenario_node in data['scenarios']:
                    self.scenarios.append(Scenario.load_yaml(scenario_node['scenario'], smalltalk_rule2grammar, text_utils))

            # INSTEAD-OF правила
            for rule in data['rules']:
                try:
                    rule = ScriptingRule.from_yaml(rule['rule'])
                    self.insteadof_rules.append(rule)
                except Exception as ex:
                    logging.error(ex)
                    raise ex

            self.smalltalk_rules = SmalltalkRules()
            self.smalltalk_rules.load_yaml(data['smalltalk_rules'], smalltalk_rule2grammar, text_utils)

            self.comprehension_rules = ComprehensionTable()
            self.comprehension_rules.load_yaml_data(data)

            self.common_phrases = []
            for common_phrase in data['common_phrases']:
                self.common_phrases.append(common_phrase)

    def get_smalltalk_rules(self):
        return self.smalltalk_rules

    def start_conversation(self, chatbot, session):
        # Начало общения с пользователем, для которого загружена сессия session
        # со всей необходимой информацией - история прежних бесед и т.д
        # Выберем одну из типовых фраз в файле smalltalk_opening.txt, вернем ее.
        logging.info(u'BotScripting::start_conversation')
        if len(self.greetings) > 0:
            return random.choice(self.greetings)

        return None

    def generate_after_answer(self, bot, answering_machine, interlocutor, interpreted_phrase, answer):
        # todo: потом вынести реализацию в производный класс, чтобы тут осталась только
        # пустая заглушка метода.

        # language_resources = answering_machine.text_utils.language_resources
        # probe_query_str = language_resources[u'как тебя зовут']
        # probe_query = InterpretedPhrase(probe_query_str)
        # answers, answer_confidenses = answering_machine.build_answers0(bot, interlocutor, probe_query)
        # ask_name = False
        # if len(answers) > 0:
        #     if answer_confidenses[0] < 0.70:
        #         ask_name = True
        # else:
        #     ask_name = True
        #
        # if ask_name:
        #     # имя собеседника неизвестно.
        #     q = language_resources[u'А как тебя зовут?']
        #     nq = answering_machine.get_session(bot, interlocutor).count_bot_phrase(q)
        #     if nq < 3:  # Не будем спрашивать более 2х раз.
        #         return q
        return None

    def get_insteadof_rules(self):
        return self.insteadof_rules

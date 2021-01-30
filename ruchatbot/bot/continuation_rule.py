"""
Правила продолжения остановившегося диалога для чит-чата
"""

import logging
import random

from ruchatbot.bot.base_rule_condition import BaseRuleCondition
from ruchatbot.utils.constant_replacer import replace_constant
from ruchatbot.bot.saying_phrase import SayingPhrase, substitute_bound_variables


class ContinuationAction:
    def __init__(self):
        self.phrases = []

    @staticmethod
    def from_yaml(yaml_node, constants, text_utils):
        actor = ContinuationAction()

        if isinstance(yaml_node, list):
            for utterance in yaml_node:
                if isinstance(utterance, str):
                    s = replace_constant(utterance, constants, text_utils)
                    actor.phrases.append(SayingPhrase(s))
                else:
                    raise SyntaxError()
        elif isinstance(yaml_node, str):
            s = replace_constant(yaml_node, constants, text_utils)
            actor.phrases.append(SayingPhrase(s))
        else:
            raise NotImplementedError()

        return actor

    def prepare4saying(self, phrase, condition_matching_results, text_utils):
        return substitute_bound_variables(phrase, condition_matching_results, text_utils)

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
        uttered_phrase = None

        # Сначала попробуем убрать из списка те реплики, которые мы уже произносили.
        new_utterances = []
        for utterance0 in self.phrases:
            utterance = self.prepare4saying(utterance0, condition_matching_results, text_utils)

            if session.count_bot_phrase(utterance) == 0:
                # Такую фразу еще не использовали
                if utterance[-1] == '?':
                    # Проверим, что бот еще не знает ответ на этот вопрос:
                    if bot.does_bot_know_answer(utterance, session, interlocutor):
                        continue

                new_utterances.append(utterance)

        if len(new_utterances) > 0:
            # Выбираем одну из оставшихся фраз.
            if len(new_utterances) == 1:
                uttered_phrase = new_utterances[0]
            else:
                uttered_phrase = random.choice(new_utterances)
        else:
            # Все фразы бот уже произнес
            pass

        return uttered_phrase


class ContinuationRule(object):
    def __init__(self):
        self.rule_name = None
        self.condition = None
        self.action = None

    @staticmethod
    def from_yaml(yaml_node, constants, text_utils):
        rule = ContinuationRule()
        if yaml_node['name']:
            rule.rule_name = yaml_node['name']
        rule.condition = BaseRuleCondition.from_yaml(yaml_node['if'], constants, text_utils)
        rule.action = ContinuationAction.from_yaml(yaml_node['then']['say'], constants, text_utils)
        return rule

    def __repr__(self):
        s = self.rule_name
        if s is None:
            s = 'ContinuationRule condition={}'.format(str(self.condition))
        return s

    def execute(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        condition_check = self.condition.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine)
        if condition_check.success:
            replica_generated = self.action.do_action(bot, session, interlocutor,
                                                      interpreted_phrase,
                                                      condition_check,
                                                      answering_engine.text_utils)
            if replica_generated:
                logging.debug('ContinuationRule "%s" outputs "%s"', self.rule_name, replica_generated)
                return  replica_generated

        return None


class ContinuationRules:
    def __init__(self):
        self.rules = []
        self.default_action = None

    def load_yaml(self, yaml_node, constants, text_utils):
        if 'rules' in yaml_node:
            for rule_yaml in yaml_node['rules']:
                rule = ContinuationRule.from_yaml(rule_yaml['rule'], constants, text_utils)
                self.rules.append(rule)

        if 'default' in yaml_node:
            self.default_action = ContinuationAction.from_yaml(yaml_node['default'], constants, text_utils)

    def generate_phrase(self, bot, session, answering_machine):
        for phrase, time_gap in session.get_interlocutor_phrases(questions=True, assertions=True, last_nb=10):
            for rule in self.rules:
                rule_res = rule.execute(bot, session, session.get_interlocutor(), phrase, answering_machine)
                if rule_res:
                    return rule_res

        if self.default_action:
            replica_generated = self.default_action.do_action(bot, session, session.get_interlocutor(),
                                                              None, None, answering_machine.text_utils)
            if replica_generated:
                logging.debug('ContinuationRules::default_phrases --> %s', replica_generated)
                return replica_generated

        return None

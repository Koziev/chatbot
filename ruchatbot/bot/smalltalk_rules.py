# -*- coding: utf-8 -*-

from abc import abstractmethod
import random
import logging

from ruchatbot.bot.base_rule_condition import BaseRuleCondition
from ruchatbot.utils.constant_replacer import replace_constant


class SmalltalkBaseCondition(object):
    def __init__(self):
        pass

    @abstractmethod
    def is_text(self):
        raise NotImplementedError()

    @abstractmethod
    def get_condition_text(self):
        raise NotImplementedError()

    @abstractmethod
    def get_key(self):
        raise NotImplementedError()


class SmalltalkTextCondition(SmalltalkBaseCondition):
    def __init__(self, condition_text):
        self.condition_text = condition_text

    def is_text(self):
        return True

    def get_condition_text(self):
        return self.condition_text

    def get_key(self):
        return u'text|'+self.condition_text


class SmalltalkComplexCondition(SmalltalkBaseCondition):
    def __init__(self, condition_yaml, constants, text_utils):
        self.condition = BaseRuleCondition.from_yaml(condition_yaml, constants, text_utils)

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        return self.condition.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine)

    def get_condition_text(self):
        raise NotImplementedError()

    def get_key(self):
        return self.condition.get_key()

    def __repr__(self):
        return self.condition.get_short_repr()


class SmalltalkBasicRule(object):
    def __init__(self, condition):
        self.condition = condition

    def get_condition_text(self):
        return self.condition.get_condition_text()

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        return self.condition.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine)

    @abstractmethod
    def is_generator(self):
        raise NotImplementedError()

    @staticmethod
    def __get_node_list(node):
        if isinstance(node, list):
            return node
        else:
            return [node]



class SmalltalkSayingRule(SmalltalkBasicRule):
    def __init__(self, name, condition):
        super(SmalltalkSayingRule, self).__init__(condition)
        self.name = name
        self.answers = []

    def get_name(self):
        return self.name

    def add_answer(self, answer):
        self.answers.append(answer)

    def is_generator(self):
        return False

    def pick_random_answer(self):
        if len(self.answers) > 1:
            return random.choise(self.answers)
        else:
            return self.answers[0]


class SmalltalkGeneratorRule(SmalltalkBasicRule):
    def __init__(self, name, condition, action_templates):
        super(SmalltalkGeneratorRule, self).__init__(condition)
        self.name = name
        self.action_templates = action_templates
        self.compiled_grammar = None

    def get_name(self):
        return self.name

    def is_generator(self):
        return True


class SmalltalkRules(object):
    def __init__(self):
        # правила с условием проверки синонимичности фраз. выносим их в отдельный
        # список, чтобы потом искать лучшее сопоставление для входной реплики одним
        # прогоном через модель синонимичности.
        self.text_rules = []

        # прочие правила с разными условиями.
        self.complex_rules = []

    @staticmethod
    def __get_node_list(node):
        if isinstance(node, list):
            return node
        else:
            return [node]

    def load_yaml(self, yaml_root, smalltalk_rule2grammar, constants, text_utils):
        """
        Загружаем список правил из yaml файла.
        yaml_root должен указывать на узел "smalltalk_rules".
        """
        for rule in yaml_root:
            condition = rule['rule']['if']
            action = rule['rule']['then']

            # Простые правила, которые задают срабатывание по тексту фразы, добавляем в отдельный
            # список, чтобы обрабатывать в модели синонимичности одним пакетом.
            if 'text' in condition and len(condition) == 1:
                for condition1 in SmalltalkRules.__get_node_list(condition['text']):
                    rule_name0 = 'smalltalk_rule_condition="{}"'.format(condition1)

                    rule_condition = SmalltalkTextCondition(replace_constant(condition1, constants, text_utils))

                    if 'say' in action:
                        rule = SmalltalkSayingRule(rule_name0, rule_condition)
                        for answer1 in SmalltalkRules.__get_node_list(action['say']):
                            answer_str = replace_constant(answer1, constants, text_utils)
                            if not isinstance(answer_str, str):
                                print('Error in rule "{}": answer "{}" is not a string!'.format(rule_name0, answer1))
                                exit(0)

                            rule.add_answer(answer_str)
                        self.text_rules.append(rule)
                    elif 'generate' in action:
                        generative_templates = list(SmalltalkRules.__get_node_list(action['generate']))
                        rule = SmalltalkGeneratorRule(rule_name0, rule_condition, generative_templates)
                        key = rule_condition.get_key()
                        if key in smalltalk_rule2grammar:
                            rule.compiled_grammar = smalltalk_rule2grammar[key]
                        else:
                            logging.error(u'Missing compiled grammar for rule %s', key)

                        self.text_rules.append(rule)
                    else:
                        logging.error(u'"%s" statement is not implemented', action)
                        raise NotImplementedError()
            else:
                rule_condition = SmalltalkComplexCondition(condition, constants, text_utils)
                rule_name0 = 'smalltalk_rule_condition="{}"'.format(str(rule_condition))
                if 'generate' in action:
                    generative_templates = list(SmalltalkRules.__get_node_list(action['generate']))
                    rule = SmalltalkGeneratorRule(rule_name0, rule_condition, generative_templates)
                    key = rule_condition.get_key()
                    if key in smalltalk_rule2grammar:
                        rule.compiled_grammar = smalltalk_rule2grammar[key]
                    else:
                        logging.error(u'Missing compiled grammar for rule "%s"', key)

                    self.complex_rules.append(rule)
                elif 'say' in action:
                    rule = SmalltalkSayingRule(rule_name0, rule_condition)
                    for answer1 in SmalltalkRules.__get_node_list(action['say']):
                        rule.add_answer(replace_constant(answer1, constants, text_utils))
                    self.complex_rules.append(rule)
                else:
                    raise NotImplementedError()

    def enumerate_text_rules(self):
        return self.text_rules

    def enumerate_complex_rules(self):
        return self.complex_rules

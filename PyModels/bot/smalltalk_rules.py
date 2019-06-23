# -*- coding: utf-8 -*-

from abc import abstractmethod


class SmalltalkBasicRule(object):
    def __init__(self, condition_text):
        self.condition_text = condition_text

    def get_condition_text(self):
        return self.condition_text

    @abstractmethod
    def is_generator(self):
        raise NotImplementedError()

    @staticmethod
    def __get_node_list(node):
        if isinstance(node, list):
            return node
        else:
            return [node]

    # @staticmethod
    # def load_yaml(rule):
    #     condition = rule['rule']['if']
    #     action = rule['rule']['then']
    #
    #     if 'text' in condition:
    #         if 'say' in action:
    #             for condition1 in SmalltalkBasicRule.__get_node_list(condition['text']):
    #                 rule11 = SmalltalkSayingRule(condition1)
    #                 for answer1 in SmalltalkBasicRule.__get_node_list(action['say']):
    #                     rule11.add_answer(answer1)
    #                 return rule11
    #         elif 'generate' in action:
    #             pass
    #
    #     raise NotImplementedError()


class SmalltalkSayingRule(SmalltalkBasicRule):
    def __init__(self, condition_text):
        super(SmalltalkSayingRule, self).__init__(condition_text)
        self.answers = []

    def add_answer(self, answer):
        self.answers.append(answer)

    def is_generator(self):
        return False


class SmalltalkGeneratorRule(SmalltalkBasicRule):
    def __init__(self, condition_text, action_templates):
        super(SmalltalkGeneratorRule, self).__init__(condition_text)
        self.action_templates = action_templates
        self.compiled_grammar = None

    def is_generator(self):
        return True

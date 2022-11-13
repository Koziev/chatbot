"""
21-01-2021 Добавлена загрузка правил в секции first_reply_rules
12.11.2022 Перенос в новую ветку
"""

import random
import logging
import yaml
import io
import os

from ruchatbot.scripting.scenario import Scenario
from ruchatbot.scripting.dialog_rule import DialogRule
from ruchatbot.utils.constant_replacer import replace_constant


class BotScripting(object):
    def __init__(self):
        self.scenarios = []  # список экземпляров класса Scenario
        self.greedy_rules = []   # жадные правила - срабатывают вместо генеративного пайплайна

    @staticmethod
    def __get_node_list(node):
        if isinstance(node, list):
            return node
        else:
            return [node]

    def load_resources(self, bot_profile, text_utils):
        if bot_profile.scenarios_enabled:
            with open(bot_profile.scenarios_path, 'r') as f:
                data = yaml.safe_load(f)
                for scenario_node in data['scenarios']:
                    scenario = Scenario.load_from_yaml(scenario_node['scenario'],
                                                       constants=bot_profile.constants,
                                                       text_utils=text_utils)
                    self.scenarios.append(scenario)
        if bot_profile.rules_enabled:
            named_patterns = dict()
            entities = dict()
            self.load_rules(bot_profile.rules_path, bot_profile, text_utils, named_patterns, entities)

    def load_rules(self, rules_path, bot_profile, text_utils, named_patterns, entities):
        with open(rules_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'import' in data:
                for path2 in data['import']:
                    import_path = os.path.join(os.path.dirname(rules_path), path2)
                    self.load_rules(import_path, bot_profile, text_utils, named_patterns, entities)

            if 'greedy_rules' in data:
                for greedy_rules_node in data['greedy_rules']:
                    rule = DialogRule.load_from_yaml(greedy_rules_node,
                                                     constants=bot_profile.constants,
                                                     named_patterns=named_patterns,
                                                     entities=entities,
                                                     text_utils=text_utils)
                    self.greedy_rules.append(rule)

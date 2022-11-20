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
from ruchatbot.scripting.scripting_module import ScriptingModule


class BotScripting(object):
    def __init__(self):
        self.scenarios = []  # список экземпляров класса Scenario
        self.named_patterns = dict()
        self.entities = []
        self.generative_named_patterns = dict()
        self.greedy_rules = []   # жадные правила - срабатывают вместо генеративного пайплайна
        self.modules = dict()  # именованные группы правил

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

                for module_node in data.get('modules', []):
                    module = ScriptingModule.load_from_yaml(module_node['module'],
                                                            modules=self.modules,
                                                            constants=bot_profile.constants,
                                                            named_patterns=self.named_patterns,
                                                            entities=self.entities,
                                                            generative_named_patterns=self.generative_named_patterns,
                                                            text_utils=text_utils)
                    self.modules[module.name] = module

                for scenario_node in data['scenarios']:
                    scenario = Scenario.load_from_yaml(scenario_node['scenario'],
                                                       modules = self.modules,
                                                       constants=bot_profile.constants,
                                                       named_patterns=self.named_patterns,
                                                       entities=self.entities,
                                                       generative_named_patterns=self.generative_named_patterns,
                                                       text_utils=text_utils)
                    self.scenarios.append(scenario)
        if bot_profile.rules_enabled:
            self.load_rules(bot_profile.rules_path, bot_profile, text_utils, self.named_patterns, self.entities, self.generative_named_patterns)

    def load_rules(self, rules_path, bot_profile, text_utils, named_patterns, entities, generative_named_patterns):
        with open(rules_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'import' in data:
                for path2 in data['import']:
                    import_path = os.path.join(os.path.dirname(rules_path), path2)
                    self.load_rules(import_path, bot_profile, text_utils, named_patterns, entities, generative_named_patterns)

            if 'greedy_rules' in data:
                for greedy_rules_node in data['greedy_rules']:
                    rule = DialogRule.load_from_yaml(greedy_rules_node,
                                                     constants=bot_profile.constants,
                                                     named_patterns=named_patterns,
                                                     entities=entities,
                                                     generative_named_patterns=generative_named_patterns,
                                                     text_utils=text_utils)
                    self.greedy_rules.append(rule)

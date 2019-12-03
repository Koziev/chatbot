# coding: utf-8

from ruchatbot.bot.actors import ActorBase
from ruchatbot.bot.smalltalk_rules import SmalltalkRules
from ruchatbot.bot.scripting_rule import ScriptingRule


class Scenario(object):
    def __init__(self):
        self.name = None
        self.priority = None
        self.on_start = None
        self.on_finish = None
        self.steps = []
        self.steps_policy = None
        self.smalltalk_rules = None
        self.insteadof_rules = None

    @staticmethod
    def load_yaml(yaml_node, smalltalk_rule2grammar, constants, text_utils):
        scenario = Scenario()
        scenario.name = yaml_node['name']
        if 'priority' in yaml_node:
            scenario.priority = int(yaml_node['priority'])
        else:
            scenario.priority = 10  # дефолтный уровень приоритета

        if 'steps_policy' in yaml_node:
            scenario.steps_policy = yaml_node['steps_policy']  # TODO - сделать проверку значения
        else:
            scenario.steps_policy = 'sequential'

        if 'steps' in yaml_node:
            for step_node in yaml_node['steps']:
                step = ActorBase.from_yaml(step_node, constants, text_utils)
                scenario.steps.append(step)

        if 'on_start' in yaml_node:
            scenario.on_start = ActorBase.from_yaml(yaml_node['on_start'], constants, text_utils)

        if 'on_finish' in yaml_node:
            scenario.on_finish = ActorBase.from_yaml(yaml_node['on_finish'], constants, text_utils)

        if 'smalltalk_rules' in yaml_node:
            scenario.smalltalk_rules = SmalltalkRules()
            scenario.smalltalk_rules.load_yaml(yaml_node['smalltalk_rules'], smalltalk_rule2grammar, constants, text_utils)

        if 'rules' in yaml_node:
            scenario.insteadof_rules = []
            for rule in yaml_node['rules']:
                rule = ScriptingRule.from_yaml(rule['rule'], constants, text_utils)
                scenario.insteadof_rules.append(rule)

        return scenario

    def get_priority(self):
        return self.priority

    def is_random_steps(self):
        return self.steps_policy == 'random'

    def is_sequential_steps(self):
        return self.steps_policy == 'sequential'

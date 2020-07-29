# coding: utf-8
"""
Сценарии - автономные фрагменты диалога.
"""

import random

from ruchatbot.bot.actors import ActorBase
from ruchatbot.bot.smalltalk_rules import SmalltalkRules
from ruchatbot.bot.scripting_rule import ScriptingRule
from ruchatbot.bot.running_scenario import RunningScenario
from ruchatbot.utils.constant_replacer import replace_constant


class ScenarioTerminationPolicy:
    def __init__(self):
        self.expiration = 0
        self.can_answer_question = None

    def load_yaml(self, yaml_node, constants, text_utils):
        if 'expiration' in yaml_node:
            self.expiration = int(yaml_node['expiration'])
        if 'can_answer_question' in yaml_node:
            self.can_answer_question = replace_constant(yaml_node['can_answer_question'], constants, text_utils)


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
        self.termination_policy = ScenarioTerminationPolicy()
        self.termination_check_count = 0

    def reset_usage_stat(self):
        """сброс статистики для целей тестирования"""
        pass

    def can_process_questions(self):
        # Обычно сценарии не обрабатывают вопросы.
        return False

    def process_question(self, running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils):
        raise NotImplementedError()

    def get_name(self):
        return self.name

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

        if 'termination_policy' in yaml_node:
            scenario.termination_policy.load_yaml(yaml_node['termination_policy'], constants, text_utils)

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

    def started(self, running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils):
        if self.on_start:
            self.on_start.do_action(bot, session, interlocutor, interpreted_phrase, condition_matching_results=None, text_utils=text_utils)

        self.run_step(running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils=text_utils)

    def run_step(self, running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils):
        # Через running_scenario передается состояние выполняющегося сценария - номер текущего шага,
        # пометки о пройденных шагах и прочая информация, которая уже не будет нужна после завершения сценария.
        assert(isinstance(running_scenario, RunningScenario))

        if self.is_sequential_steps():
            # Шаги сценария выполняются в заданном порядке
            while True:
                new_step_index = running_scenario.current_step_index + 1
                if new_step_index == len(running_scenario.scenario.steps):
                    # Сценарий исчерпан
                    if self.check_termination(bot, session, interlocutor, interpreted_phrase, text_utils):
                        # Уберем инстанс сценария из списка активных
                        self.termination_check_count = 0
                        if running_scenario.scenario.on_finish:
                            running_scenario.scenario.on_finish.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils)
                        bot.get_engine().exit_scenario(bot, session, interlocutor, interpreted_phrase)
                    break
                else:
                    running_scenario.current_step_index = new_step_index
                    step = running_scenario.scenario.steps[new_step_index]
                    running_scenario.passed_steps.add(new_step_index)
                    step_ok = step.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils)
                    if step_ok:
                        break

        elif running_scenario.scenario.is_random_steps():
            # Шаги сценария выбираются в рандомном порядке, не более 1 раза каждый шаг.
            nsteps = len(running_scenario.scenario.steps)
            step_indeces = list(i for i in range(nsteps) if i not in running_scenario.passed_steps)
            new_step_index = random.choice(step_indeces)
            running_scenario.passed_steps.add(new_step_index)
            step = running_scenario.scenario.steps[new_step_index]
            step.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils=text_utils)

            if len(running_scenario.passed_steps) == nsteps:
                # Больше шагов нет
                if self.check_termination(bot, session, interlocutor, interpreted_phrase, text_utils):
                    # Уберем инстанс сценария из списка активных
                    self.termination_check_count = 0
                    if running_scenario.scenario.on_finish:
                        running_scenario.scenario.on_finish.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils=text_utils)
                    bot.get_engine().exit_scenario(bot, session, interlocutor, interpreted_phrase)
        else:
            raise NotImplementedError()

    def check_termination(self, bot, session, interlocutor, interpreted_phrase, text_utils):
        """Вернет True, если сценарий надо выключать"""
        self.termination_check_count += 1
        if self.termination_check_count >= self.termination_policy.expiration:
            return True

        if self.termination_policy.can_answer_question:
            f = bot.does_bot_know_answer(self.termination_policy.can_answer_question, session, interlocutor)
            if f:
                return True

        return False

    def get_insteadof_rules(self):
        return self.insteadof_rules
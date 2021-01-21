# coding: utf-8
"""
Сценарии - автономные, долгоживущие фрагменты диалога, выполняющие управление дискурсом.

20.01.2021 Реализация графовых сценариев с вероятностными условными переходами
"""

import random
import logging

from ruchatbot.bot.actors import ActorBase, ActorSay
from ruchatbot.bot.base_rule_condition import BaseRuleCondition

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


class ScenarioTransition:
    def __init__(self):
        self.condition = None
        #self.proba = None
        self.next_step = None

    @staticmethod
    def load_yaml(yaml_node, constants, text_utils):
        t = ScenarioTransition()
        t.next_step = yaml_node['goto']
        if 'keyword' in yaml_node or 'text' in yaml_node:
            t.condition = BaseRuleCondition.from_yaml(yaml_node, constants, text_utils)
        else:
            raise RuntimeError()
        return t


class ScenarioStep:
    def __init__(self):
        self.name = None
        self.next_step_names = None
        self.action = None
        self.transitions = []

    @staticmethod
    def load_yaml(yaml_node, constants, text_utils):
        step = ScenarioStep()
        step.name = yaml_node['name']
        step.say = ActorSay.from_yaml(yaml_node['say'], constants, text_utils)

        if 'default_next' in yaml_node:
            if isinstance(yaml_node['default_next'], str):
                step.next_step_names = [yaml_node['default_next']]
            else:
                step.next_step_names = list(yaml_node['default_next'])

        if 'transitions' in yaml_node:
            for tr_yaml in yaml_node['transitions']:
                step.transitions.append(ScenarioTransition.load_yaml(tr_yaml['transition'], constants, text_utils))

        return step

    @staticmethod
    def from_say_actor(step_index, step_node, constants, text_utils):
        step = ScenarioStep()
        step.name = str(step_index)
        step.say = ActorBase.from_yaml(step_node, constants, text_utils)
        return step

    def get_name(self):
        return self.name


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
    def load_yaml(yaml_node, global_bot_scripting, smalltalk_rule2grammar, constants, text_utils):
        scenario = Scenario()
        scenario.name = yaml_node['name']
        if 'priority' in yaml_node:
            scenario.priority = int(yaml_node['priority'])
        else:
            scenario.priority = 10  # дефолтный уровень приоритета

        if 'steps_policy' in yaml_node:
            scenario.steps_policy = yaml_node['steps_policy']
            if scenario.steps_policy not in 'sequential random graf'.split():
                raise RuntimeError('Scenario "{}" loading error: unknown step_policy "{}"'.format(scenario.name, scenario.steps_policy))
        else:
            scenario.steps_policy = 'sequential'

        if 'chitchat_questions_per_step_rate' in yaml_node:
            scenario.chitchat_questions_per_step_rate = yaml_node['chitchat_questions_per_step_rate']
        else:
            scenario.chitchat_questions_per_step_rate = 0

        if 'steps' in yaml_node:
            if scenario.steps_policy == 'graf':
                for step_node in yaml_node['steps']:
                    step = ScenarioStep.load_yaml(step_node['step'], constants, text_utils)
                    scenario.steps.append(step)
            else:
                for step_node in yaml_node['steps']:
                    step = ScenarioStep.from_say_actor(len(scenario.steps), step_node, constants, text_utils)
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

        scenario.insteadof_rules = []
        if 'rules' in yaml_node:
            for rule in yaml_node['rules']:
                rule = ScriptingRule.from_yaml(rule['rule'], constants, text_utils)
                scenario.insteadof_rules.append(rule)

        if 'insteadof_rule_import' in yaml_node:
            insteadof_rule_import = yaml_node['insteadof_rule_import']
            if insteadof_rule_import == 'from_global':
                # добавляем в список глобальные insteadof-правила
                scenario.insteadof_rules.extend(global_bot_scripting.insteadof_rules)

        return scenario

    def get_priority(self):
        return self.priority

    def is_random_steps(self):
        return self.steps_policy == 'random'

    def is_sequential_steps(self):
        return self.steps_policy == 'sequential'

    def is_graf(self):
        return self.steps_policy == 'graf'

    def get_chitchat_questions_per_step_rate(self):
        return self.chitchat_questions_per_step_rate

    def started(self, running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils):
        if self.on_start:
            self.on_start.do_action(bot, session, interlocutor, interpreted_phrase,
                                    condition_matching_results=None,
                                    text_utils=text_utils)

        #self.run_step(running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils=text_utils)
        return

    def run_step(self, running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils):
        # Через running_scenario передается состояние выполняющегося сценария - номер текущего шага,
        # пометки о пройденных шагах и прочая информация, которая уже не будет нужна после завершения сценария.
        assert(isinstance(running_scenario, RunningScenario))

        if self.is_graf():
            next_step_name = None
            transition_found = False

            # Смотрим не текущий узел в графе
            if running_scenario.current_step_index == -1:
                # Это вход в граф. Потом надо сделать более гибкий механизм поиска стартового узла,
                # а сейчас просто возьмем первый в списке и выполним его (он скажет что-то.)
                start_index = 0  # !!! TODO !!!
                next_step_name = self.steps[start_index].get_name()
                transition_found = True
                logging.debug('Entering scenario "%s" to node "%s"', self.get_name(), next_step_name)
            else:
                current_step = self.steps[running_scenario.current_step_index]
                logging.debug('Continuing scenario "%s" from node "%s"', self.get_name(), current_step.get_name())

                # У него есть условные переходы?
                if current_step.transitions:
                    # Проверяем, не должны ли мы перейти по одному из этих условных переходов.
                    for tr in current_step.transitions:
                        if tr.condition.check_condition(bot, session, interlocutor, interpreted_phrase, bot.get_engine()):
                            next_step_name = tr.next_step
                            break

                if next_step_name is None and current_step.next_step_names:
                    # Дефолтный переход на один из указанных узлов
                    next_step_name = random.choice(current_step.next_step_names)

            # Ищем этот узел в графе ...
            if next_step_name and next_step_name != '_EXIT_':
                found = False
                for inode, node in enumerate(self.steps):
                    if node.get_name() == next_step_name:
                        # Нашли!
                        found = True
                        running_scenario.current_step_index = inode
                        logging.debug('Transition to node "%s" in scenario "%s"', next_step_name, self.get_name())
                        transition_found = True
                        step_ok = node.say.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils)
                        break

                if not found:
                    logging.error('Node "%s" not found in scenario "%s"', next_step_name, self.get_name())

            if not transition_found:
                # По умолчанию - выходим из сценария...
                logging.debug('Dead-end in scenario "%s"', self.get_name())

                # Уберем инстанс сценария из списка активных
                self.termination_check_count = 0
                if running_scenario.scenario.on_finish:
                    running_scenario.scenario.on_finish.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils=text_utils)
                bot.get_engine().exit_scenario(bot, session, interlocutor, interpreted_phrase)

        elif self.is_sequential_steps():
            # Шаги сценария выполняются в заданном порядке
            while True:
                new_step_index = running_scenario.current_step_index + 1
                if new_step_index == len(running_scenario.scenario.steps):
                    # Сценарий исчерпан
                    logging.debug('Scenario "%s" is exhausted', self.get_name())

                    if self.check_termination(bot, session, interlocutor, interpreted_phrase, text_utils):
                        # Уберем инстанс сценария из списка активных
                        self.termination_check_count = 0
                        if running_scenario.scenario.on_finish:
                            running_scenario.scenario.on_finish.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils)
                        bot.get_engine().exit_scenario(bot, session, interlocutor, interpreted_phrase)
                    break
                else:
                    running_scenario.current_step_index = new_step_index
                    logging.debug('Executing step #%d in scenario "%s"', new_step_index, self.get_name())
                    step = running_scenario.scenario.steps[new_step_index]
                    running_scenario.passed_steps.add(new_step_index)
                    step_ok = step.say.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils)
                    if step_ok:
                        break

        elif running_scenario.scenario.is_random_steps():
            # Шаги сценария выбираются в рандомном порядке, не более 1 раза каждый шаг.
            nsteps = len(running_scenario.scenario.steps)
            step_indeces = list(i for i in range(nsteps) if i not in running_scenario.passed_steps)
            new_step_index = random.choice(step_indeces)
            logging.debug('Executing step #%d in scenario "%s"', new_step_index, self.get_name())
            running_scenario.passed_steps.add(new_step_index)
            step = running_scenario.scenario.steps[new_step_index]
            step.say.do_action(bot, session, interlocutor, interpreted_phrase, None, text_utils=text_utils)

            do_terminate = False
            if len(running_scenario.passed_steps) == nsteps:
                # Больше шагов нет
                do_terminate = True
            elif self.check_termination(bot, session, interlocutor, interpreted_phrase, text_utils):
                do_terminate = True

            if do_terminate:
                logging.debug('Scenario "%s" is exhausted', self.get_name())

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

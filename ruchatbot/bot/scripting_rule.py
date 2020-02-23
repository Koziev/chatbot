# -*- coding: utf-8 -*-

import logging

from abc import abstractmethod
from ruchatbot.bot.actors import ActorBase
from ruchatbot.bot.base_rule_condition import BaseRuleCondition


class ScriptingRuleResult(object):
    def __init__(self):
        self.replica_is_generated = None
        self.condition_success = False

    @staticmethod
    def unmatched():
        return ScriptingRuleResult()

    @staticmethod
    def matched(replica_is_generated):
        res = ScriptingRuleResult()
        res.condition_success = True
        res.replica_is_generated = replica_is_generated
        return res


class ScriptingRule(object):
    def __init__(self):
        pass

    @staticmethod
    def from_yaml(yaml_node, constants, text_utils):
        if 'if' in yaml_node:
            rule = ScriptingRuleIf(yaml_node, constants, text_utils)
            return rule
        elif 'switch' in yaml_node:
            rule = ScriptingRuleSwitch(yaml_node, constants, text_utils)
            return rule
        else:
            raise NotImplementedError()

    @abstractmethod
    def execute(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        raise NotImplementedError()


class ScriptingRuleIf(ScriptingRule):
    def __init__(self, yaml_node, constants, text_utils):
        self.condition = BaseRuleCondition.from_yaml(yaml_node['if'], constants, text_utils)
        self.compiled_action = ActorBase.from_yaml(yaml_node['then'], constants, text_utils)

        if 'name' in yaml_node:
            self.rule_name = yaml_node['name']
        else:
            self.rule_name = self.condition.get_short_repr()

        if 'priority' in yaml_node:
            self.priority = float(yaml_node['priority'])
        else:
            self.priority = 1.0

    def __repr__(self):
        s = self.rule_name
        if s is None:
            s = 'ScriptingRuleIf condition={}'.format(str(self.condition))
        return s

    def execute(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        """Вернет True, если правило сформировало ответную реплику."""
        condition_check = self.condition.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine)
        if condition_check.success:
            logging.debug('ScriptingRuleIf "%s"', self.rule_name)
            session.rule_activated(self)
            replica_generated = self.compiled_action.do_action(bot, session, interlocutor, interpreted_phrase,
                                                               condition_check, answering_engine.text_utils)
            return ScriptingRuleResult.matched(replica_generated)
        else:
            return ScriptingRuleResult.unmatched()


class ScriptingRuleSwitch(ScriptingRule):
    def __init__(self, yaml_node, constants, text_utils):
        self.rule_name = None
        if 'name' in yaml_node:
            self.rule_name = yaml_node['name']

        if 'priority' in yaml_node:
            self.priority = float(yaml_node['priority'])
        else:
            self.priority = 1.0

        self.condition1 = BaseRuleCondition.from_yaml(yaml_node['switch']['when'], constants, text_utils)
        self.case_handlers = []
        self.default_handler = None
        cases = yaml_node['switch']['cases']
        for answer_case in cases:
            answer_case = answer_case['case']
            if 'if' in answer_case:
                case_handler = ScriptingRuleIf(answer_case, constants, text_utils)
                self.case_handlers.append(case_handler)
            else:
                raise NotImplementedError()

        if 'default' in yaml_node['switch']:
            self.default_handler = ActorBase.from_yaml(yaml_node['switch']['default'], constants, text_utils)

    def __repr__(self):
        if self.rule_name:
            return self.rule_name
        else:
            return str(self.condition1)


    def execute(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        condition_check = self.condition1.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine)
        if condition_check.success:
            # Главное условие (проверка заданного вопроса) успешно проверено, теперь ищем подходящий
            # обработчик для ответа.
            for case_handler in self.case_handlers:
                assert(isinstance(case_handler, ScriptingRuleIf))
                case_check = case_handler.condition.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine)
                if case_check.success:
                    logging.debug('ScriptingRuleSwitch "%s" ==> case_handler.do_action', self.rule_name)
                    replica_generated = case_handler.compiled_action.do_action(bot, session, interlocutor,
                                                                               interpreted_phrase, case_check, answering_engine.text_utils)
                    return ScriptingRuleResult.matched(replica_generated)

            if self.default_handler:
                # Ни один из обработчиков не отработал, запускаем default-обработчик
                logging.debug('ScriptingRuleSwitch "%s" ==> default_handler.do_action', self.rule_name)
                replica_generated = self.default_handler.do_action(bot, session, interlocutor, interpreted_phrase,
                                                                   None, answering_engine.text_utils)
                return ScriptingRuleResult.matched(replica_generated)

        return ScriptingRuleResult.unmatched()

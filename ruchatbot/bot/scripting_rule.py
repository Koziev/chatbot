# -*- coding: utf-8 -*-

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
    def from_yaml(yaml_node):
        if 'if' in yaml_node:
            condition = yaml_node['if']
            action = yaml_node['then']
            rule = ScriptingRuleIf(condition, action)
            return rule
        elif 'switch' in yaml_node:
            rule = ScriptingRuleSwitch(yaml_node)
            return rule
        else:
            raise NotImplementedError()

    @abstractmethod
    def execute(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        raise NotImplementedError()


class ScriptingRuleIf(ScriptingRule):
    def __init__(self, condition, action):
        self.condition = BaseRuleCondition.from_yaml(condition)
        self.action = action
        self.compiled_action = ActorBase.from_yaml(action)

    def execute(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        """Вернет True, если правило сформировало ответную реплику."""
        if self.condition.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine):
            session.rule_activated(self)
            replica_generated = self.compiled_action.do_action(bot, session, interlocutor, interpreted_phrase)
            return ScriptingRuleResult.matched(replica_generated)
        else:
            return ScriptingRuleResult.unmatched()


class ScriptingRuleSwitch(ScriptingRule):
    def __init__(self, yaml_node):
        self.condition1 = BaseRuleCondition.from_yaml(yaml_node['switch']['when'])
        self.case_handlers = []
        self.default_handler = None
        cases = yaml_node['switch']['cases']
        for answer_case in cases:
            answer_case = answer_case['case']
            if 'if' in answer_case:
                case_handler = ScriptingRuleIf(answer_case['if'], answer_case['then'])
                self.case_handlers.append(case_handler)
            else:
                raise NotImplementedError()

        if 'default' in yaml_node['switch']:
            self.default_handler = ActorBase.from_yaml(yaml_node['switch']['default'])

    def execute(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        if self.condition1.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine):
            # Главное условие (проверка заданного вопроса) успешно проверено, теперь ищем подходящий
            # обработчик для ответа.
            for case_handler in self.case_handlers:
                assert(isinstance(case_handler, ScriptingRuleIf))
                if case_handler.condition.check_condition(bot, session, interlocutor, interpreted_phrase, answering_engine):
                    replica_generated = case_handler.compiled_action.do_action(bot, session, interlocutor, interpreted_phrase)
                    return ScriptingRuleResult.matched(replica_generated)

            if self.default_handler:
                # Ни один из обработчиков не отработал, запускаем default-обработчик
                replica_generated = self.default_handler.do_action(bot, session, interlocutor, interpreted_phrase)
                return ScriptingRuleResult.matched(replica_generated)

        return ScriptingRuleResult.unmatched()

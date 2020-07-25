# -*- coding: utf-8 -*-

import uuid
import logging


class BotPersonality:
    """Предполагается, что этот класс будет хранить модели и правила, определяющие
    поведение экземпляра бота и его характер. Базовые модели типа детектора
    синонимичности и релевантности хранятся в общем для всех экземпляров движке
    чатбота (SimpleAnsweringMachine)"""
    def __init__(self, bot_id, engine, facts, profile, faq=None, scripting=None):
        if bot_id is None or bot_id == '':
            self.bot_id = str(uuid.uuid4())
        else:
            self.bot_id = bot_id

        self.engine = engine
        self.facts = facts
        self.faq = faq
        self.scripting = scripting
        self.profile = profile
        self.enable_scripting = profile.rules_enabled
        self.enable_smalltalk = profile.smalltalk_enabled
        self.force_question_answering = profile.force_question_answering
        self.premise_is_answer = profile.premise_is_answer
        self.replica_after_answering = profile.replica_after_answering
        self.generative_smalltalk_enabled = profile.generative_smalltalk_enabled

        self.event_handlers = dict()
        self.on_process_order = None

    def get_bot_id(self):
        return self.bot_id

    def has_scripting(self):
        return self.scripting is not None and self.enable_scripting

    def get_scripting(self):
        return self.scripting

    def get_engine(self):
        return self.engine

    def extract_entity(self, entity_name, interpreted_phrase):
        return self.extract_entity_from_str(entity_name, interpreted_phrase.interpretation)

    def extract_entity_from_str(self, entity_name, phrase_str):
        return self.engine.extract_entity(entity_name, phrase_str)

    def get_comprehension_templates(self):
        return self.scripting.comprehension_rules

    def start_conversation(self, user_id):
        self.engine.start_conversation(self, user_id)

    def pop_phrase(self, user_id):
        # todo переделка
        return self.engine.pop_phrase(self, user_id)

    def cancel_all_running_items(self, user_id):
        self.engine.cancel_all_running_items(self, user_id)

    def reset_session(self, user_id):
        self.engine.reset_session(self, user_id)

    def reset_usage_stat(self):
        if self.scripting:
            self.scripting.reset_usage_stat()
        self.engine.reset_usage_stat()

    def reset_added_facts(self):
        self.facts.reset_added_facts()

    def push_phrase(self, user_id, question, internal_issuer=False):
        self.engine.push_phrase(self, user_id, question, internal_issuer,
                                force_question_answering=self.force_question_answering)

    def process_order(self, session, user_id, interpreted_phrase):
        order_str = interpreted_phrase.interpretation
        if self.on_process_order is not None:
            return self.on_process_order(order_str, self, session)
        else:
            return False

    #def apply_insteadof_rule(self, session, interlocutor, interpreted_phrase):
    #    return False

    def say(self, session, phrase):
        self.engine.say(self, session, phrase)

    def add_event_handler(self, event_name, handler):
        self.event_handlers[event_name] = handler

    def invoke_callback(self, event_name, session, user_id, interpreted_phrase, verb_form_fields):
        if self.event_handlers is not None:
            if event_name in self.event_handlers:
                return self.event_handlers[event_name](self, session, user_id, interpreted_phrase, verb_form_fields)
            elif u'*' in self.event_handlers:
                return self.event_handlers[u'*'](event_name, self, session, user_id, interpreted_phrase, verb_form_fields)
            else:
                logging.error(u'No handler for callback event "{}"'.format(event_name))

    def run_form(self, form_actor, session, user_id, interpreted_phrase):
        for form in self.scripting.forms:
            if form.name == form_actor.form_name:
                self.engine.run_form(form, self, session, user_id, interpreted_phrase)
                return

        raise KeyError(form_actor.form_name)

    def run_scenario(self, scenario_actor, session, user_id, interpreted_phrase):
        for scenario in self.scripting.scenarios:
            if scenario.name == scenario_actor.scenario_name:
                if scenario_actor.mode == 'replace':
                    self.engine.run_scenario(scenario, self, session, user_id, interpreted_phrase)
                elif scenario_actor.mode == 'call':
                    self.engine.call_scenario(scenario, self, session, user_id, interpreted_phrase)
                else:
                    raise NotImplementedError()
                return

        raise KeyError(scenario_actor.scenario_name)

    def get_common_phrases(self):
        return self.scripting.common_phrases

    def does_bot_know_answer(self, question, session, interlocutor):
        return self.engine.does_bot_know_answer(question, self, session, interlocutor)

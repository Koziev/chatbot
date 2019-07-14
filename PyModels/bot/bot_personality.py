# -*- coding: utf-8 -*-

import uuid
import logging


class BotPersonality:
    """Предполагается, что этот класс будет хранить модели и правила, определяющие
    поведение экземпляра бота и его характер. Базовые модели типа детектора
    синонимичности и релевантности хранятся в общем для всех экземпляров движке
    чатбота (SimpleAnsweringMachine)"""
    def __init__(self, bot_id, engine, facts,
                 faq=None, scripting=None,
                 enable_smalltalk=False,
                 enable_scripting=False,
                 force_question_answering=False):
        if bot_id is None or bot_id == '':
            self.bot_id = str(uuid.uuid4())
        else:
            self.bot_id = bot_id

        self.engine = engine
        self.facts = facts
        self.faq = faq
        self.scripting = scripting
        self.enable_smalltalk = enable_smalltalk
        self.enable_scripting = enable_scripting
        self.premise_is_answer = False
        self.event_handlers = dict()
        self.on_process_order = None
        self.force_question_answering = force_question_answering

    def get_bot_id(self):
        return self.bot_id

    def has_scripting(self):
        return self.scripting is not None and self.enable_scripting

    def get_scripting(self):
        return self.scripting

    def get_engine(self):
        return self.engine

    def extract_entity(self, entity_name, phrase):
        return self.engine.extract_entity(entity_name, phrase.interpretation)

    def get_comprehension_templates(self):
        return self.scripting.comprehension_rules

    def start_conversation(self, user_id):
        self.engine.start_conversation(self, user_id)

    def pop_phrase(self, user_id):
        # todo переделка
        return self.engine.pop_phrase(self, user_id)

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
        self.engine.say(session, phrase)

    def add_event_handler(self, event_name, handler):
        self.event_handlers[event_name] = handler

    def invoke_callback(self, event_name, session, user_id, interpreted_phrase):
        if self.event_handlers is not None:
            if event_name in self.event_handlers:
                return self.event_handlers[event_name](self, session, user_id, interpreted_phrase)
            elif u'*' in self.event_handlers:
                return self.event_handlers[u'*'](event_name, self, session, user_id, interpreted_phrase)
            else:
                logging.error(u'No handler for callback event "{}"'.format(event_name))

    def get_common_phrases(self):
        return self.scripting.common_phrases

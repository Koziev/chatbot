# -*- coding: utf-8 -*-

import uuid


class BotPersonality:
    def __init__(self, bot_id, engine, facts, scripting=None,
                 enable_smalltalk=False, enable_scripting=False):
        if bot_id is None or bot_id == '':
            self.bot_id = str(uuid.uuid4())
        else:
            self.bot_id = bot_id

        self.engine = engine
        self.facts = facts
        self.scripting = scripting
        self.enable_smalltalk = enable_smalltalk
        self.enable_scripting = enable_scripting
        self.premise_is_answer = False
        self.order_templates = None
        self.on_process_order = None

    def get_bot_id(self):
        return self.bot_id

    def has_scripting(self):
        return self.scripting is not None and self.enable_scripting

    def set_order_templates(self, order_templates):
        self.order_templates = order_templates

    def start_conversation(self, user_id):
        self.engine.start_conversation(self, user_id)

    def pop_phrase(self, user_id):
        # todo переделка
        return self.engine.pop_phrase(self, user_id)

    def push_phrase(self, user_id, question):
        self.engine.push_phrase(self, user_id, question)

    def process_order(self, session, interpreted_phrase):
        order_str = interpreted_phrase.interpretation
        if self.on_process_order is not None:
            self.on_process_order(order_str, self, session)

    def say(self, session, phrase):
        self.engine.say(session, phrase)

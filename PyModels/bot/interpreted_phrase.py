# -*- coding: utf-8 -*-

class InterpretedPhrase:
    def __init__(self, raw_phrase):
        self.is_bot_phrase = False
        self.raw_phrase = raw_phrase
        self.interpretation = raw_phrase
        self.is_question = None
        self.phrase_person = None

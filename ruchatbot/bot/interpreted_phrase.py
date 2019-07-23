# -*- coding: utf-8 -*-

from ruchatbot.bot.modality_detector import ModalityDetector


class InterpretedPhrase:
    def __init__(self, raw_phrase):
        self.is_bot_phrase = False
        self.raw_phrase = raw_phrase
        self.interpretation = raw_phrase
        self.is_question = None
        self.is_imperative = None
        self.intent = None
        self.person = None

    def set_modality(self, modality, person):
        self.person = person
        if modality == ModalityDetector.imperative:
            self.is_imperative = True
            self.is_question = False
        elif modality == ModalityDetector.question:
            self.is_question = True
            self.is_imperative = False
        else:
            self.is_question = False
            self.is_imperative = False

    @property
    def is_assertion(self):
        return not self.is_question and not self.is_imperative

    def __repr__(self):
        return u'{} (raw="{}" intent="{}")'.format(self.interpretation, self.raw_phrase, self.intent)

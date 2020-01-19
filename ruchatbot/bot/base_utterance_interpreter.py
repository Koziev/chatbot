# -*- coding: utf-8 -*-

from abc import abstractmethod

from ruchatbot.bot.model_applicator import ModelApplicator


class BaseUtteranceInterpreter(ModelApplicator):
    def __init__(self):
        pass

    @abstractmethod
    def require_interpretation(self, phrase, text_utils):
        raise NotImplementedError()

    @abstractmethod
    def interpret(self, phrases, text_utils):
        raise NotImplementedError()

    @abstractmethod
    def normalize_person(self, raw_phrase, text_utils):
        raise NotImplementedError()

    @abstractmethod
    def denormalize_person(self, normal_phrase, text_utils):
        raise NotImplementedError()

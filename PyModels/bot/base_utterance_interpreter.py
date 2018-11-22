# -*- coding: utf-8 -*-

from model_applicator import ModelApplicator


class BaseUtteranceInterpreter(ModelApplicator):
    def __init__(self):
        pass

    def require_interpretation(self, phrase, text_utils, word_embeddings):
        raise NotImplemented()

    def interpret(self, phrases, text_utils, word_embeddings):
        raise NotImplemented()


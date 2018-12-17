# -*- coding: utf-8 -*-

from model_applicator import ModelApplicator


class ModalityDetector(ModelApplicator):
    """
    Интерфейс для детекторов модальности фраз собеседника - спрашивает ли он,
    отвечат или приказывает.
    """
    question = 'question'
    assertion = 'assertion'
    imperative = 'imperative'
    undefined = 'undefined'

    def __init__(self):
        pass

    def get_modality(self, phrase, text_utils, word_embeddings):
        raise NotImplementedError()

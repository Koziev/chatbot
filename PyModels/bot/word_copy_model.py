# -*- coding: utf-8 -*-

from model_applicator import ModelApplicator


class WordCopyModel(ModelApplicator):
    def __init__(self):
        pass

    def generate_answer(self, premise_str, question_str, text_utils, word_embeddings):
        raise NotImplementedError()

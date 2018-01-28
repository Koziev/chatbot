# -*- coding: utf-8 -*-

import os
from model_applicator import ModelApplicator

class YesNoModel(ModelApplicator):
    def __init__(self):
        pass

    def calc_yes_no(self, premise_words, question_words, text_utils, word_embeddings):
        raise NotImplemented()

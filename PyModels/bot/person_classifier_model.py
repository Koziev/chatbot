# -*- coding: utf-8 -*-

import os
from model_applicator import ModelApplicator

class PersonClassifierModel(ModelApplicator):
    def __init__(self):
        pass

    def detect_person(self, sentence_str, text_utils, word_embeddings):
        raise NotImplemented()

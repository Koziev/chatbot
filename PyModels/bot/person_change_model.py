# -*- coding: utf-8 -*-

import os
from model_applicator import ModelApplicator


class PersonChangeModel(ModelApplicator):
    def __init__(self):
        pass

    def change_person(self, sentence_str, target_person, text_utils, word_embeddings):
        raise NotImplemented()

# -*- coding: utf-8 -*-


class Base_FAQ_Bot(object):
    def __init__(self, text_utils):
        self.text_utils

    def load_models(self, models_folder, w2v_folder):
        raise NotImplementedError()

    def select_answer(self, question, text_utils):
        raise NotImplementedError()

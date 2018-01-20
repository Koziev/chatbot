# -*- coding: utf-8 -*-

import os

class YesNoModel(object):
    def __init__(self):
        pass

    def load(self, models_folder):
        """
        Производный класс должен загрузить в этом методе все свои файлы
        с данными и конфигурацией.
        :param models_folder: папка, где располагаются все файлы модели.
        """
        raise NotImplemented()

    def calc_yes_no(self, premise, question, text_utils, word_embeddings):
        raise NotImplemented()

    def get_model_filepath(self, models_folder, old_filepath):
        _, tail = os.path.split(old_filepath)
        return os.path.join( models_folder,  tail )

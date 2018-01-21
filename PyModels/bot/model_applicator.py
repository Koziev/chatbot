# -*- coding: utf-8 -*-

import os

class ModelApplicator(object):
    """
    Класс предназначен для скрытия деталей применения различных
    моделей в движке чат-бота.
    """
    def __init__(self):
        pass

    def load(self, models_folder):
        """
        Производный класс должен загрузить в этом методе все свои файлы
        с данными и конфигурацией.
        :param models_folder: папка, где располагаются все файлы модели.
        :type arg1: unicode
        """
        raise NotImplemented()

    def get_model_filepath(self, models_folder, old_filepath):
        """
        Из полного пути 'old_filepath' вырезаем имя файла, добавляем к нему
        заданную папку 'models_folder' и возвращаем новый полный путь
        :param models_folder: папка, в которой фактически располагаются файлы моделей
        :type arg1: unicode
        :param old_filepath: путь к файлу данных, сохраненный тренером модели
        :type arg2: unicode
        :return: актуальный путь к файлу данных
        :rtype: unicode
        """
        _, tail = os.path.split(old_filepath)
        return os.path.join( models_folder,  tail )

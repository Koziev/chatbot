# -*- coding: utf-8 -*-

"""
Модель для генерации ответа в ситуации, когда релевантная предпосылка не найдена.
Сейчас модель минимальна и просто возвращает предопределенный ответ "нет информации" либо
одну из фраз в специальном файле. В будущем тут можно либо сделать обращение
к внешнему сервису, либо генерировать реплики генеративной моделью.
"""

import io
import os
import random
import logging

from model_applicator import ModelApplicator


class NoInformationModel(ModelApplicator):
    def __init__(self):
        super(NoInformationModel, self).__init__()
        self.replicas = []

    def load(self, model_folder, data_folder):
        path = os.path.join(data_folder, 'no_information_replicas.txt')
        logging.info(u'Loading replicas from {}'.format(path))
        with io.open(path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                replica = line.strip()
                if replica:
                    self.replicas.append(replica)
        logging.info(u'{} replicas loaded'.format(len(self.replicas)))

    def generate_answer(self, phrase, bot, text_utils, word_embeddings):
        if len(self.replicas) > 1:
            return random.choice(self.replicas)
        else:
            return self.replicas[0]

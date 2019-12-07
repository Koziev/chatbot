# -*- coding: utf-8 -*-

"""
Модель для генерации ответа в ситуации, когда релевантная предпосылка не найдена либо
приказ не удалось интерпретировать.

Сейчас модель минимальна и просто возвращает предопределенный ответ "нет информации" либо
одну из фраз в специальном файле. В будущем тут можно либо сделать обращение
к внешнему сервису, либо генерировать реплики генеративной моделью.
"""

import io
import os
import random
import logging
import yaml

from ruchatbot.bot.model_applicator import ModelApplicator
from ruchatbot.bot.scripting_rule import ScriptingRule


class NoInformationModel(ModelApplicator):
    """
    Класс инкапсулирует генерацию ответов в двух специальных случаях: 1) когда для ответа
    не удалось подобрать предпосылку или найти правило обработки 2) когда не найдено
    правило для обработки приказа.

    Сейчас модель просто выбирает одну из фраз, прописанных в конфигурационном файле rules.yaml.
    """
    def __init__(self):
        super(NoInformationModel, self).__init__()
        self.no_info_replicas = []
        self.unknown_order = []
        self.rules = []

    def load(self, model_folder, data_folder, constants, text_utils):
        yaml_path = os.path.join(data_folder, 'rules.yaml')
        logging.info(u'Loading NoInformationModel replicas from "%s"', yaml_path)

        with io.open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.no_info_replicas = data['no_relevant_information']['phrases']
            self.unknown_order = data['unknown_order']

            if 'rules' in data['no_relevant_information']:
                for rule_yaml in data['no_relevant_information']['rules']:
                    rule = ScriptingRule.from_yaml(rule_yaml['rule'], constants, text_utils)
                    self.rules.append(rule)

        logging.info(u'NoInformationModel loaded: %d phrase(s), %d rule(s)', len(self.no_info_replicas), len(self.rules))

    def get_noanswer_rules(self):
        return self.rules

    def generate_answer(self, phrase, bot, text_utils, word_embeddings):
        if len(self.no_info_replicas) > 1:
            return random.choice(self.no_info_replicas)
        else:
            return self.replicas[0]

    def order_not_understood(self, phrase, bot, text_utils, word_embeddings):
        if len(self.unknown_order) > 1:
            return random.choice(self.unknown_order)
        else:
            return self.unknown_order[0]

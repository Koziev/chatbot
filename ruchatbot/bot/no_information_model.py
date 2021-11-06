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
from ruchatbot.utils.constant_replacer import replace_constant


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
        self.no_info_replicas2 = []
        self.no_info_replicas3 = []
        self.unknown_order = []
        self.rules = []

    def load(self, rule_paths, model_folder, data_folder, constants, text_utils):
        for yaml_path in rule_paths:
            logging.info('Loading NoInformationModel replicas and rules from "%s"', yaml_path)

            with io.open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if 'no_relevant_information' in data:
                    y = data['no_relevant_information']
                    if 'phrases' in y:
                        for s in y['phrases']:
                            self.no_info_replicas.append(replace_constant(s, constants, text_utils))

                    if 'phrases2' in y:
                        for s in y['phrases2']:
                            self.no_info_replicas2.append(replace_constant(s, constants, text_utils))

                    if 'phrases3' in y:
                        for s in y['phrases3']:
                            self.no_info_replicas3.append(replace_constant(s, constants, text_utils))

                    if 'unknown_order' in y:
                        for s in y['unknown_order']:
                            self.unknown_order.append(replace_constant(s, constants, text_utils))

                    if 'rules' in y:
                        for rule_yaml in y['rules']:
                            rule = ScriptingRule.from_yaml(rule_yaml['rule'], constants, text_utils)
                            self.rules.append(rule)

        logging.info('NoInformationModel loaded: %d phrase(s), %d rule(s)', len(self.no_info_replicas), len(self.rules))

    def get_noanswer_rules(self):
        return self.rules

    def generate_answer(self, phrase, bot, session, text_utils):
        a = None
        if session.cannot_answer_counter == 1:
            if len(self.no_info_replicas2) > 1:
                a = random.choice(self.no_info_replicas2)
        elif session.cannot_answer_counter == 2:
            if len(self.no_info_replicas3) > 1:
                a = random.choice(self.no_info_replicas3)

        if a is None:
            if len(self.no_info_replicas) > 1:
                a = random.choice(self.no_info_replicas)
            elif len(self.no_info_replicas) == 1:
                a = self.no_info_replicas[0]

        session.cannot_answer_counter += 1
        return a

    def order_not_understood(self, phrase, bot, session, text_utils):
        s = None
        if len(self.unknown_order) > 0:
            s = random.choice(self.unknown_order)

        return s

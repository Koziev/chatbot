# -*- coding: utf-8 -*-
"""
Базовый класс для реализации вариантов детектора релевантности,
использующих градиентный бустинг и разреженные матрицы шинглов.
См. реализацию класса XGB_RelevancyDetector
"""

import logging

from ruchatbot.bot.relevancy_detector import RelevancyDetector
from ruchatbot.bot.gb_base_detector import GB_BaseDetector


class GB_RelevancyDetector(RelevancyDetector):
    def __init__(self):
        super(GB_RelevancyDetector, self).__init__()
        self.logger = logging.getLogger('GB_RelevancyDetector')
        self.engine = None

        # для XGBoost тип элементов входной матрицы может быть bool,
        # для LightGBM должен быть 'float32'
        self.x_matrix_type = '<<unknown>>'

    def predict_by_model(self, X_data):
        """
        Метод должен быть переопределен в классе-потомке и реализовывать вызов расчета релевантностей
        для матрицы X_Data, содержащей фичи для пар вопрос-предпосылка.
        """
        raise NotImplementedError()

    def init_model_params(self, model_config):
        self.engine = GB_BaseDetector()
        self.engine.x_matrix_type = self.x_matrix_type
        self.engine.init_model_params(model_config)

    def calc_relevancy1(self, premise, question, text_utils):
        return self.engine.calc_relevancy1(premise, question, text_utils,
                                           predictor_func=lambda X_data: self.predict_by_model(X_data))

    def get_most_relevant(self, probe_phrase, phrases, text_utils, nb_results=1):
        return self.engine.get_most_relevant(probe_phrase, phrases, text_utils,
                                             predictor_func=lambda X_data: self.predict_by_model(X_data),
                                             nb_results=nb_results)

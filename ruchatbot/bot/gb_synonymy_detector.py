# -*- coding: utf-8 -*-
"""
Базовый класс для реализации вариантов детектора синонимичности,
использующих градиентный бустинг и разреженные матрицы шинглов.
См. реализацию класса XGB_RelevancyDetector
"""

import logging

from ruchatbot.bot.synonymy_detector import SynonymyDetector
from ruchatbot.bot.gb_base_detector import GB_BaseDetector


class GB_SynonymyDetector(SynonymyDetector):
    def __init__(self):
        super(GB_SynonymyDetector, self).__init__()
        self.logger = logging.getLogger('GB_SynonymyDetector')
        self.engine = None

        # для XGBoost тип элементов входной матрицы может быть bool,
        # для LightGBM должен быть 'float32'
        self.x_matrix_type = '<<unknown>>'

    def predict_by_model(self, X_data):
        """
        Метод должен быть переопределен в классе-потомке и реализовывать вызом расчета релевантностей
        для матрицы X_Data, содержащей фичи для пар вопрос-предпосылка.
        """
        raise NotImplementedError()

    def init_model_params(self, model_config):
        self.engine = GB_BaseDetector()
        self.engine.x_matrix_type = self.x_matrix_type
        self.engine.init_model_params(model_config)

    def get_most_similar(self, probe_phrase, phrases, text_utils, word_embeddings, nb_results=1):
        return self.engine.get_most_relevant(probe_phrase,
                                             phrases,
                                             text_utils,
                                             predictor_func=lambda X_data: self.predict_by_model(X_data),
                                             nb_results=nb_results)

    def calc_synonymy2(self, phrase1, phrase2, text_utils, word_embeddings):
        return self.engine.calc_relevancy1(phrase1, phrase2,
                                           text_utils,
                                           predictor_func=lambda X_data: self.predict_by_model(X_data))

# -*- coding: utf-8 -*-

import json
import os
import logging
import xgboost

from gb_relevancy_detector import GB_RelevancyDetector


class XGB_RelevancyDetector(GB_RelevancyDetector):
    """
    Модель для определения релевантности предпосылки и вопроса на базе
    классификатора XGBoost.
    """
    def __init__(self):
        super(XGB_RelevancyDetector, self).__init__()
        self.logger = logging.getLogger('XGB_RelevancyDetector')
        self.x_matrix_type = 'bool'

    def load(self, models_folder):
        self.logger.info('Loading XGB_RelevancyDetector model files')
        # Определение релевантности предпосылки и вопроса на основе XGB модели
        with open(os.path.join(models_folder, 'xgb_relevancy.config'), 'r') as f:
            model_config = json.load(f)

        self.init_model_params(model_config)

        self.xgb_relevancy = xgboost.Booster()
        self.xgb_relevancy.load_model(self.get_model_filepath(models_folder,  model_config['model_filename']))

    def predict_by_model(self, X_data):
        D_data = xgboost.DMatrix(X_data)
        y_probe = self.xgb_relevancy.predict(D_data)
        return y_probe

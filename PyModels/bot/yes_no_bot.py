# -*- coding: utf-8 -*-

import logging

from lgb_relevancy_detector import LGB_RelevancyDetector
from xgb_yes_no_model import XGB_YesNoModel


class YesNoBot:
    def __init__(self, text_utils):
        self.text_utils = text_utils
        self.logger = logging.getLogger('YesNoBot')
        self.relevancy_detector = LGB_RelevancyDetector()
        self.yes_no_model = XGB_YesNoModel()
        self.word_embeddings = None

    def load_models(self, models_folder):
        self.logger.info(u'Loading models from {}'.format(models_folder))
        self.models_folder = models_folder
        self.relevancy_detector.load(models_folder)
        self.yes_no_model.load(models_folder)

    def get_yes_answer(self):
        return u'да'

    def get_no_answer(self):
        return u'нет'

    def get_unknown_answer(self):
        return u'неопределено'

    def infer_answer(self, premises0, question0):
        question = self.text_utils.canonize_text(question0)
        premises = [self.text_utils.canonize_text(f) for f in premises0]

        if len(premises) > 1:
            self.logger.error(u'{} premises input is not supported'.format(len(premises)))
            return u''
        else:
            premise = premises[0]
            rel = self.relevancy_detector.calc_relevancy1(premise, question, self.text_utils, self.word_embeddings)
            self.logger.debug('relevancy={}'.format(rel))
            if rel < 0.5:
                return self.get_unknown_answer()

            y = self.yes_no_model.calc_yes_no(premise, question, self.text_utils, self.word_embeddings)
            self.logger.debug('y={}'.format(y))
            if y < 0.5:
                return self.get_no_answer()

            return self.get_yes_answer()






# -*- coding: utf-8 -*-

import logging
import os

from lgb_relevancy_detector import LGB_RelevancyDetector
from xgb_yes_no_model import XGB_YesNoModel
from nn_yes_no_model import NN_YesNoModel
from word_embeddings import WordEmbeddings


class YesNoBot:
    def __init__(self, text_utils):
        self.text_utils = text_utils
        self.logger = logging.getLogger('YesNoBot')
        self.relevancy_detector = LGB_RelevancyDetector()
        #self.yes_no_model = XGB_YesNoModel()
        self.yes_no_model = NN_YesNoModel()
        self.word_embeddings = WordEmbeddings()
        self.show_relevancy = True

    def load_models(self, models_folder, w2v_folder):
        self.logger.info(u'Loading models from {}'.format(models_folder))
        self.models_folder = models_folder
        self.relevancy_detector.load(models_folder)
        self.yes_no_model.load(models_folder)

        self.wordchar2vector_path = os.path.join(models_folder, 'wordchar2vector.dat')

        self.word_embeddings.load_wc2v_model(self.wordchar2vector_path)
        p = self.yes_no_model.w2v_path
        if p is not None:
            p = os.path.join(w2v_folder, os.path.basename(p))
            self.word_embeddings.load_w2v_model(p)


    def get_yes_answer(self):
        return self.text_utils.language_resources[u'да']

    def get_no_answer(self):
        return self.text_utils.language_resources[u'нет']

    def get_unknown_answer(self):
        return self.text_utils.language_resources[u'неопределено']

    def infer_answer(self, premises0, question0):
        premises = [self.text_utils.canonize_text(f) for f in premises0]
        question = self.text_utils.canonize_text(question0)

        if len(premises) > 1:
            self.logger.error(u'{} premises input is not supported'.format(len(premises)))
            return u''
        else:
            if False:
                if len(premises) == 1:
                    # Проверим, что введенная пользователем предпосылка релевантна заданному вопросу.
                    premise = premises[0]
                    rel = self.relevancy_detector.calc_relevancy1(premise, question, self.text_utils, self.word_embeddings)
                    self.logger.debug('relevancy={}'.format(rel))
                    if rel < 0.5:
                        if self.show_relevancy:
                            return self.get_unknown_answer() + ' ({})'.format(rel)
                        else:
                            return self.get_unknown_answer()

            y = self.yes_no_model.calc_yes_no(premises, question, self.text_utils, self.word_embeddings)
            self.logger.debug('y={}'.format(y))
            if y < 0.5:
                return self.get_no_answer()

            return self.get_yes_answer()

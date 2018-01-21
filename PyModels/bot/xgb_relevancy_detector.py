# -*- coding: utf-8 -*-

import xgboost
from scipy.sparse import lil_matrix
import json
import os
import logging

from relevancy_detector import RelevancyDetector


class XGB_RelevancyDetector(RelevancyDetector):
    """
    Модель для определения релевантности предпосылки и вопроса на базе XGBoost.
    """
    def __init__(self):
        super(XGB_RelevancyDetector, self).__init__()
        self.logger = logging.getLogger('XGB_RelevancyDetector')

    def unknown_shingle(self, shingle):
        self.logger.error(u'Shingle "{}" is unknown'.format(shingle))

    def load(self, models_folder):
        self.logger.info('Loading XGB_RelevancyDetector model files')
        # Определение релевантности предпосылки и вопроса на основе XGB модели
        with open(os.path.join(models_folder, 'xgb_relevancy.config'), 'r') as f:
            model_config = json.load(f)

        self.xgb_relevancy_shingle2id = model_config['shingle2id']
        self.xgb_relevancy_shingle_len = model_config['shingle_len']
        self.xgb_relevancy_nb_features = model_config['nb_features']
        self.xgb_relevancy = xgboost.Booster()
        self.xgb_relevancy.load_model( self.get_model_filepath( models_folder,  model_config['model_filename'] ) )

    def get_most_relevant(self, probe_phrase, phrases, text_utils, word_embeddings):
        nb_answers = len(phrases)

        # Поиск наиболее релевантной предпосылки с помощью XGB модели
        X_data = lil_matrix((nb_answers, self.xgb_relevancy_nb_features), dtype='bool')

        # все предпосылки из текущей базы фактов векторизуем в один тензор, чтобы
        # прогнать его через классификатор разом.
        best_premise = ''
        best_sim = 0.0
        for ipremise, (premise, premise_person, phrase_code) in enumerate(phrases):
            premise_words = text_utils.tokenize(premise)
            question_words = text_utils.tokenize(probe_phrase)
            premise_wx = text_utils.words2str(premise_words)
            question_wx = text_utils.words2str(question_words)

            premise_shingles = set(text_utils.ngrams(premise_wx, self.xgb_relevancy_shingle_len))
            question_shingles = set(text_utils.ngrams(question_wx, self.xgb_relevancy_shingle_len))

            self.xgb_relevancy_vectorize_sample_x(X_data, ipremise, premise_shingles, question_shingles, self.xgb_relevancy_shingle2id)

        D_data = xgboost.DMatrix(X_data)
        y_probe = self.xgb_relevancy.predict(D_data)

        reslist = []
        for ipremise, (premise, premise_person, phrase_code) in enumerate(phrases):
            sim = y_probe[ipremise]
            reslist.append( (premise, sim) )

        reslist = sorted(reslist, key=lambda z: -z[1])

        best_premise = reslist[0][0]
        best_rel = reslist[0][1]
        return best_premise, best_rel


    def xgb_relevancy_vectorize_sample_x(self, X_data, idata, premise_shingles, question_shingles, shingle2id):
        ps = set(premise_shingles)
        qs = set(question_shingles)
        common_shingles = ps & qs
        notmatched_ps = ps - qs
        notmatched_qs = qs - ps

        nb_shingles = len(shingle2id)

        icol = 0
        for shingle in common_shingles:
            if shingle not in shingle2id:
                self.unknown_shingle(shingle)
            else:
                X_data[idata, icol + shingle2id[shingle]] = True

        icol += nb_shingles
        for shingle in notmatched_ps:
            if shingle not in shingle2id:
                self.unknown_shingle(shingle)
            else:
                X_data[idata, icol + shingle2id[shingle]] = True

        icol += nb_shingles
        for shingle in notmatched_qs:
            if shingle not in shingle2id:
                self.unknown_shingle(shingle)
            else:
                X_data[idata, icol + shingle2id[shingle]] = True

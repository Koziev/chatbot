import os
import json
import numpy as np
import logging
import pickle

import scipy
import lightgbm


class P2Q_Relevancy_LGB:
    def __init__(self):
        super(P2Q_Relevancy_LGB, self).__init__()
        self.logger = logging.getLogger('P2Q_Relevancy_LGB')
        self.model = None
        self.model_config = None
        self.vectorizer = None

    def load(self, models_folder):
        self.logger.info('Loading P2Q_Relevancy_LGB model files')

        with open(os.path.join(models_folder, '2premises_question_relevancy_via_lgb.config'), 'r') as f:
            self.model_config = json.load(f)

        model_path = os.path.join(models_folder, os.path.basename(self.model_config['model_filename']))
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        vectorizer_path = os.path.join(models_folder, os.path.basename(self.model_config['vectorizer_filename']))
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

    def calc_relevancy(self, premise1, premise2, question, text_utils):
        premise1 = text_utils.remove_terminators(premise1.strip())
        words = text_utils.tokenizer.tokenize(premise1)
        premise1 = u' '.join(words)

        premise2 = text_utils.remove_terminators(premise2.strip())
        words = text_utils.tokenizer.tokenize(premise2)
        premise2 = u' '.join(words)

        question = text_utils.remove_terminators(question.strip())
        words = text_utils.tokenizer.tokenize(question)
        question = u' '.join(words)

        X1 = self.vectorizer.transform([premise1])
        X2 = self.vectorizer.transform([premise2])
        X3 = self.vectorizer.transform([question])

        x = self.model_config['x']
        if x == 'x1*x3;x2*x3':
            x13 = X1.copy()
            x13.multiply(X3)

            x23 = X2.copy()
            x23.multiply(X3)

            X = scipy.sparse.hstack((x13, x23))

        elif x == 'x1*x3;x2*x3;x1;x2;x3':
            x13 = X1.copy()
            x13.multiply(X3)

            x23 = X2.copy()
            x23.multiply(X3)

            X = scipy.sparse.hstack((x13, x23, X1, X2, X3))

        elif x == 'x1*x3;x2*x3;x1*x2':
            x13 = X1.copy()
            x13.multiply(X3)

            x23 = X2.copy()
            x23.multiply(X3)

            x12 = X1.copy()
            x12.multiply(X2)

            X = scipy.sparse.hstack((x13, x23, x12))

        elif x == 'x1*x3;x2*x3;x1*x2;x1;x2;x3':
            x13 = X1.copy()
            x13.multiply(X3)

            x23 = X2.copy()
            x23.multiply(X3)

            x12 = X1.copy()
            x12.multiply(X2)

            X = scipy.sparse.hstack((x13, x23, x12, X1, X2, X3))

        y_pred = self.model.predict_proba(X)
        y_pred = y_pred[0][1]
        return y_pred

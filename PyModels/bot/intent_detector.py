# coding: utf-8

from __future__ import print_function
import pickle
import os

# TODO: сделать базовый класс
class IntentDetector(object):
    def __init__(self):
        pass

    def load(self, model_dir):
        model_path = os.path.join(model_dir, 'intent_classifier.model')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def detect_intent(self, phrase_str):
        X_query = self.model['vectorizer'].transform([phrase_str])
        y_query = self.model['estimator'].predict(X_query)
        intent_index = y_query[0]
        intent_name = self.model['index2label'][intent_index]
        return intent_name



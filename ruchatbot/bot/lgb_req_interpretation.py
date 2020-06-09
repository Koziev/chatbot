# -*- coding: utf-8 -*-
"""
Реализация классификатора интерпретируемости фраз на LightGBM модели bag-of-shingles
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

import os
import json
import numpy as np
import logging
import pickle

import lightgbm

from ruchatbot.bot.base_utterance_interpreter import BaseUtteranceInterpreter
from ruchatbot.utils.padding_utils import PAD_WORD, lpad_wordseq, rpad_wordseq


class LGB_ReqInterpretation(BaseUtteranceInterpreter):
    def __init__(self):
        super(LGB_ReqInterpretation, self).__init__()
        self.logger = logging.getLogger('LGB_ReqInterpretation')
        self.model = None
        self.vectorizer = None
        self.model_config = None
        self.no_expansion_phrases = set()

    def load(self, models_folder):
        self.logger.info('Loading LGB_ReqInterpretation model files')

        with open(os.path.join(models_folder, 'lgb_req_interpretation.config'), 'r') as f:
            self.model_config = json.load(f)

        model_path = os.path.join(models_folder, os.path.basename(self.model_config['model_filename']))
        self.model = lightgbm.Booster(model_file=self.model_config['model_filename'])

        vectorizer_path = os.path.join(models_folder, os.path.basename(self.model_config['vectorizer_filename']))
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        self.no_expansion_phrases = set(self.model_config['no_expansion_phrases'])

    def require_interpretation(self, phrase0, text_utils):
        phrase = text_utils.remove_terminators(phrase0.strip())
        phrase_words = text_utils.tokenizer.tokenize(phrase)
        phrase_str = u' '.join(phrase_words)

        if phrase_str in self.no_expansion_phrases:
            return False

        X = self.vectorizer.transform([phrase_str])
        y_pred = self.model.predict(X)
        y_pred = y_pred[0]
        return y_pred > 0.5

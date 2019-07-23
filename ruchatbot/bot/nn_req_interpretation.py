# -*- coding: utf-8 -*-
"""
Нейросетевая реализация модели интерпретации реплик собеседника.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

import os
import json
import numpy as np
import logging

from keras.models import model_from_json

from ruchatbot.bot.base_utterance_interpreter import BaseUtteranceInterpreter
from ruchatbot.utils.padding_utils import PAD_WORD, lpad_wordseq, rpad_wordseq


class NN_ReqInterpretation(BaseUtteranceInterpreter):
    def __init__(self):
        super(NN_ReqInterpretation, self).__init__()
        self.logger = logging.getLogger('NN_ReqInterpretation')
        self.model = None
        self.model_config = None

    def load(self, models_folder):
        self.logger.info('Loading NN_ReqInterpretation model files')

        arch_filepath = os.path.join(models_folder, 'nn_req_interpretation.arch')
        weights_path = os.path.join(models_folder, 'nn_req_interpretation.weights')
        with open(arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(weights_path)
        self.model = m

        with open(os.path.join(models_folder, 'nn_req_interpretation.config'), 'r') as f:
            self.model_config = json.load(f)

        self.word_dims = self.model_config['word_dims']
        self.w2v_path = self.model_config['w2v_path']
        self.padding = self.model_config['padding']
        self.max_wordseq_len = self.model_config['max_wordseq_len']
        self.w2v_filename = os.path.basename(self.w2v_path)

    def pad_wordseq(self, words, n):
        if self.padding == 'left':
            return lpad_wordseq(words, n)
        else:
            return rpad_wordseq(words, n)

    def require_interpretation(self, phrase0, text_utils, word_embeddings):
        phrase = text_utils.remove_terminators(phrase0.strip())
        phrase_words = text_utils.tokenizer.tokenize(phrase)

        X_batch  = np.zeros((1, self.max_wordseq_len, self.word_dims), dtype=np.float32)

        words = self.pad_wordseq(phrase_words, self.max_wordseq_len)
        word_embeddings.vectorize_words(self.w2v_filename, words, X_batch, 0)

        y_pred = self.model.predict(x=X_batch, verbose=0)
        y_pred = y_pred[0]
        return y_pred[1] > 0.5

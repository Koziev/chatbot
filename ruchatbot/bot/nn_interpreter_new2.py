# -*- coding: utf-8 -*-
"""
Реализация новой модели интерпретации реплик собеседника,
в том числе заполнение пропусков (гэппинг, эллипспс), нормализация грамматического лица.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.

07-06-2020 Полная переделка на новую модель интерпретации (seq2seq with attention)
"""

import os
import json
import numpy as np
import logging
import random
import pickle
import itertools

from keras.models import model_from_json

#import keras_contrib
#from keras_contrib.layers import CRF
#from keras_contrib.losses import crf_loss
#from keras_contrib.metrics import crf_viterbi_accuracy

import sentencepiece as spm

# https://github.com/asmekal/keras-monotonic-attention
from ruchatbot.layers.attention_decoder import AttentionDecoder

from ruchatbot.bot.base_utterance_interpreter2 import BaseUtteranceInterpreter2


class Sample(object):
    def __init__(self, context_phrases, short_phrase):
        self.context_phrases = context_phrases
        self.short_phrase = short_phrase


class NN_InterpreterNew2(BaseUtteranceInterpreter2):
    def __init__(self):
        super(NN_InterpreterNew2, self).__init__()
        self.logger = logging.getLogger('NN_InterpreterNew2')
        self.model = None
        self.model_config = None
        self.bpe_model = None
        self.index2token = None
        self.token2index = None
        self.seq_len = None

    def load(self, models_folder):
        self.logger.info('Loading NN_InterpreterNew2 model files')

        # Файлы нейросетевой модели интерпретации
        with open(os.path.join(models_folder, 'nn_seq2seq_interpreter.config'), 'r') as f:
            self.model_config = json.load(f)

            arch_file = os.path.join(models_folder, os.path.basename(self.model_config['arch_path']))
            weights_file = os.path.join(models_folder, os.path.basename(self.model_config['weights_path']))
            bpe_model_name = self.model_config['bpe_model_name']

            with open(arch_file, 'r') as f:
                self.model = model_from_json(f.read(), {'AttentionDecoder': AttentionDecoder})

            self.model.load_weights(weights_file)

            self.bpe_model = spm.SentencePieceProcessor()
            rc = self.bpe_model.Load(os.path.join(models_folder, bpe_model_name + '.model'))
            assert(rc is True)

            self.index2token = dict((i, t) for t, i in self.model_config['token2index'].items())
            self.token2index = self.model_config['token2index']
            self.seq_len = self.model_config['max_left_len']

        #self.interpret_pointer_words = set((u'твой твоя твое твои твоего твоей твоим твоими твоих твоем твоему твоей ' +
        #                                u'мой моя мое мои моего моей моих моими моим моем моему').split())

        super(NN_InterpreterNew2, self).load(models_folder)

    def vectorize_samples(self, samples, text_utils):
        nb_samples = len(samples)
        X1 = np.zeros((nb_samples, self.seq_len), dtype=np.int32)

        for isample, sample in enumerate(samples):
            left_phrases = [text_utils.wordize_text(s) for s in sample.context_phrases]
            left_phrases.append(text_utils.wordize_text(sample.short_phrase))
            left_str = ' | '.join(left_phrases)
            left_tokens = self.bpe_model.EncodeAsPieces(left_str)

            for itoken, token in enumerate(left_tokens[:self.seq_len]):
                if token in self.token2index:
                    X1[isample, itoken] = self.token2index[token]

        return [X1]

    def interpret(self, phrases, text_utils, generative_grammar):
        if len(phrases) < 2:
            logging.warning('%d input phrase(s) in NN_InterpreterNew2::interpret, at least 2 expected', len(phrases))
            return phrases[-1]

        context_phrases = phrases[:-1]
        short_phrase = phrases[-1]

        samples = [Sample(context_phrases, short_phrase)]
        X_data = self.vectorize_samples(samples, text_utils)

        y_pred = self.model.predict(x=X_data, verbose=0)
        y_pred = np.argmax(y_pred[0], axis=-1)
        tokens = [self.index2token[itok] for itok in y_pred]
        expanded_phrase = ''.join(tokens).replace('▁', ' ').strip()

        self.logger.debug('NN_InterpreterNew2 expanded_phrase="%s"', expanded_phrase)
        return expanded_phrase

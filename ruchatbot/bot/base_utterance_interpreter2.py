# -*- coding: utf-8 -*-
"""
Реализация методов нормализации и денормализации лица для интерпретатора.
"""

import os
import json
import re
import numpy as np
import logging
import random
import pickle
import itertools

from keras.models import model_from_json

import keras_contrib
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from ruchatbot.bot.base_utterance_interpreter import BaseUtteranceInterpreter
from ruchatbot.utils.padding_utils import PAD_WORD, lpad_wordseq, rpad_wordseq


class BaseUtteranceInterpreter2(BaseUtteranceInterpreter):
    def __init__(self):
        super(BaseUtteranceInterpreter2, self).__init__()
        self.logger = logging.getLogger('BaseUtteranceInterpreter2')

    def load(self, models_folder):
        self.logger.info('Loading BaseUtteranceInterpreter2 model files')

        # Таблицы для трансляции грамматического лица
        with open(os.path.join(models_folder, 'person_change_dictionary.pickle'), 'rb') as f:
            self.person_changing_data = pickle.load(f)

        #self.w1s = self.person_changing_data['word_1s']
        #self.w2s = self.person_changing_data['word_2s']
        self.person_change_1s_2s = self.person_changing_data['person_change_1s_2s']
        self.person_change_2s_1s = self.person_changing_data['person_change_2s_1s']

        self.hard_replacement = {u'я': u'ты',
                                 u'ты': u'я'}

        self.special_changes_3 = {u'меня': u'тебя',
                                  u'мне': u'тебе',
                                  u'мной': u'тобой',
                                  u'мною': u'тобою',
                                  u'тебя': u'меня',
                                  u'тебе': u'мне',
                                  u'тобой': u'мной',
                                  u'тобою': u'мною',
                                  u'по-моему': u'по-твоему',
                                  u'по-твоему': u'по-моему',
                                  u'ваш': u'мой',
                                  u'наш': u'ваш',
                                  u'по-вашему': u'по-моему'
                                  }

    def flip_person(self, src_phrase, text_utils):
        inwords = text_utils.tokenize(src_phrase)
        outwords = []
        for word in inwords:
            if word in self.hard_replacement:
                outwords.append(self.hard_replacement[word])
            else:
                if word in self.person_change_1s_2s:
                    outwords.append(self.person_change_1s_2s[word])
                elif word in self.person_change_2s_1s:
                    outwords.append(self.person_change_2s_1s[word])
                else:
                    # немного хардкода.
                    if word in self.special_changes_3:
                        outwords.append(self.special_changes_3[word])
                    else:
                        outwords.append(word)

        return u' '.join(outwords)

    def postprocess_prepositions(self, s):
        s = re.sub(r'\bко тебе\b', 'к тебе', s)
        s = re.sub(r'\bобо тебе\b', 'о тебе', s)
        s = re.sub(r'\bсо тобой\b', 'с тобой', s)
        s = re.sub(r'\bво тебе\b', 'в тебе', s)

        s = re.sub(r'\bк мне\b', 'ко мне', s)
        s = re.sub(r'\bо мне\b', 'обо мне', s)
        s = re.sub(r'\bс мной\b', 'со мне', s)
        s = re.sub(r'\bв мне\b', 'во мне', s)
        return s

    def normalize_person(self, raw_phrase, text_utils):
        return self.postprocess_prepositions(self.flip_person(raw_phrase, text_utils))

    def denormalize_person(self, normal_phrase, text_utils):
        return self.postprocess_prepositions(self.flip_person(normal_phrase, text_utils))

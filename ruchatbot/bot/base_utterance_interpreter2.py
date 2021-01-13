# -*- coding: utf-8 -*-
"""
Реализация методов нормализации и денормализации лица для интерпретатора.

12.01.2021 Добавляем работу с репликами в уважительной форме 2л мн.ч "как Вас зовут?"
"""

import os
import re
import logging
import pickle

from ruchatbot.bot.base_utterance_interpreter import BaseUtteranceInterpreter


class BaseUtteranceInterpreter2(BaseUtteranceInterpreter):
    def __init__(self):
        super(BaseUtteranceInterpreter2, self).__init__()
        self.logger = logging.getLogger('BaseUtteranceInterpreter2')

    def load(self, models_folder):
        self.logger.info('Loading BaseUtteranceInterpreter2 model files')

        # Таблицы для трансляции грамматического лица
        with open(os.path.join(models_folder, 'person_change_dictionary.pickle'), 'rb') as f:
            self.person_changing_data = pickle.load(f)

        self.person_change_1s_2s = self.person_changing_data['person_change_1s_2s']
        self.person_change_2s_1s = self.person_changing_data['person_change_2s_1s']
        self.person_change_2p_1s = self.person_changing_data['person_change_2p_1s']

        self.hard_replacement = {'я': 'ты',
                                 'ты': 'я',
                                 'вы': 'я'}

        self.special_changes_3 = {'меня': 'тебя',
                                  'мне': 'тебе',
                                  'мной': 'тобой',
                                  'мною': 'тобою',

                                  'тебя': 'меня',
                                  'тебе': 'мне',
                                  'тобой': 'мной',
                                  'тобою': 'мною',

                                  'вас': 'меня',
                                  'вам': 'мне',
                                  'вами': 'мной',

                                  'по-моему': 'по-твоему',
                                  'по-твоему': 'по-моему',
                                  'по-вашему': 'по-моему',

                                  'ваш': 'мой',
                                  'ваши': 'мои',
                                  'вашим': 'моим',
                                  'вашими': 'моими',
                                  'ваших': 'моих',
                                  'вашем': 'моем',
                                  'вашему': 'моему',
                                  'вашей': 'моей',
                                  'вашу': 'мою',
                                  'ваша': 'моя',
                                  'ваше': 'мое',

                                  'твой': 'мой',
                                  'твои': 'мои',
                                  'твоим': 'моим',
                                  'твоими': 'моими',
                                  'твоих': 'моих',
                                  'твоем': 'моем',
                                  'твоему': 'моему',
                                  'твоей': 'моей',
                                  'твою': 'мою',
                                  'твоя': 'моя',
                                  'твое': 'мое',

                                  'наш': 'ваш',
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
                elif word in self.person_change_2p_1s:
                    outwords.append(self.person_change_2p_1s[word])
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
        s = re.sub(r'\bс мной\b', 'со мной', s)
        s = re.sub(r'\bв мне\b', 'во мне', s)
        return s

    def normalize_person(self, raw_phrase, text_utils):
        return self.postprocess_prepositions(self.flip_person(raw_phrase, text_utils))

    def denormalize_person(self, normal_phrase, text_utils):
        return self.postprocess_prepositions(self.flip_person(normal_phrase, text_utils))

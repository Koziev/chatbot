# -*- coding: utf-8 -*-
"""
Реализация методов нормализации и денормализации лица для интерпретатора.

12.01.2021 Добавляем работу с репликами в уважительной форме 2л мн.ч "как Вас зовут?"
03.10.2021 Удаляем пробелы перед некоторыми знаками пунктуации
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
            uword = word.lower()
            is_Aa = uword != word

            new_word = word
            if uword in self.hard_replacement:
                new_word = self.hard_replacement[uword]
            else:
                if uword in self.person_change_1s_2s:
                    new_word = self.person_change_1s_2s[uword]
                elif uword in self.person_change_2s_1s:
                    new_word = self.person_change_2s_1s[uword]
                elif uword in self.person_change_2p_1s:
                    new_word = self.person_change_2p_1s[uword]
                else:
                    # немного хардкода.
                    if uword in self.special_changes_3:
                        new_word = self.special_changes_3[uword]

            if new_word != word and is_Aa:
                new_word = new_word[0].upper() + new_word[1:]

            outwords.append(new_word)

        return self.normalize_delimiters(' '.join(outwords))

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

    def normalize_delimiters(self, s):
        return s.replace(' ?', '?').replace(' ,', ',').replace(' .', '.').replace(' !', '!')

    def normalize_person(self, raw_phrase, text_utils):
        return self.normalize_delimiters(self.postprocess_prepositions(self.flip_person(raw_phrase, text_utils)))

    def denormalize_person(self, normal_phrase, text_utils):
        return self.normalize_delimiters(self.postprocess_prepositions(self.flip_person(normal_phrase, text_utils)))

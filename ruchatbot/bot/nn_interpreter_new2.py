# -*- coding: utf-8 -*-
"""
Реализация новой модели интерпретации реплик собеседника,
в том числе заполнение пропусков (гэппинг, эллипсис), нормализация грамматического лица.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.

07-06-2020 Полная переделка на новую модель интерпретации (seq2seq with attention)
20-06-2020 Добавка шаблонной модели knn-1
28-07-2020 Исправление ошибки с потерей tf-сессии
"""

import os
import json
import numpy as np
import logging
import random
import pickle
import itertools

import tensorflow as tf
from keras.models import model_from_json

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
        self.templates = None
        self.graph = None

    def load(self, models_folder):
        self.logger.info('Loading NN_InterpreterNew2 model files')

        self.graph = tf.get_default_graph()

        # Эталонные экземпляры для knn-1 модели
        with open(os.path.join(models_folder, 'interpreter_templates2.bin'), 'rb') as f:
            self.templates = pickle.load(f)

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

    def is_important_token2(self, t):
        pos = t[1].split('|')[0]
        if pos in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'ADP', 'PREP'):
            return True

        lemma = t[2]
        if lemma in ('да', 'нет', 'не', 'ни', 'ага'):
            return True

        return False

    def match_template1(self, template1, context1):
        if len(template1) == len(context1):
            match1 = dict()
            for template_item, token in zip(template1, context1):
                if template_item[1] is None:
                    # проверка формы слова, этот токен не используется для подстановки в развернутую фразу
                    if template_item[0] != token[0]:
                        return None
                else:
                    # проверяем грамматические теги, запоминаем лемму для последующей вставки в развернутую форму
                    if all((tag in token[1]) for tag in template_item[1]):
                        match1[template_item[2]] = token[2]
                    else:
                        return None

            return match1

        return None

    def match_template(self, template, context):
        match = dict()
        all_matched = True
        if len(template) == len(context):  # совпадает кол-во проверяемых фраз контекста
            for template1, context1 in zip(template, context):
                match1 = self.match_template1(template1, context1)
                if match1:
                    match.update(match1)
                else:
                    all_matched = False
                    break

        return match if all_matched else None

    def prepare_context_line(self, line, text_utils):
        tokens = text_utils.lemmatize2(line)
        tokens = [(t[0], t[1].split('|'), t[2]) for t in tokens if self.is_important_token2(t)]
        return tokens

    def generate_output_by_template(self, output_template, matching, text_utils):
        res_words = []
        for word, location, tags in output_template:
            if word is not None:
                res_words.append(word)
            else:
                lemma = matching[location]
                all_tags = dict(tags[1:])
                required_tags = ''
                if tags[0] == 'NOUN':
                    required_tags = 'ПАДЕЖ ЧИСЛО'.split()
                elif tags[0] == 'ADJ':
                    required_tags = 'РОД ПАДЕЖ ЧИСЛО ОДУШ СТЕПЕНЬ'.split()
                elif tags[0] == 'VERB':
                    required_tags = 'ВРЕМЯ ЛИЦО ЧИСЛО РОД НАКЛОНЕНИЕ'.split()
                elif tags[0] == 'ADV':
                    required_tags = 'СТЕПЕНЬ'

                required_tags = [(t, all_tags[t]) for t in required_tags if t in all_tags]
                if required_tags:
                    forms = list(text_utils.flexer.find_forms_by_tags(lemma, required_tags))
                    if forms:
                        form = forms[0]
                    else:
                        form = lemma
                else:
                    # сюда попадает случай инфинитива
                    form = lemma

                res_words.append(form)

        return ' '.join(res_words)

    def interpret(self, phrases, text_utils):
        if len(phrases) < 2:
            logging.warning('%d input phrase(s) in NN_InterpreterNew2::interpret, at least 2 expected', len(phrases))
            return phrases[-1]

        context_phrases = phrases[:-1]
        short_phrase = phrases[-1]

        expanded_phrase = None

        # Сначала пробуем knn-1 модель, ищем подходящий шаблон
        context2 = [self.prepare_context_line(s, text_utils) for s in phrases]
        for it, template in enumerate(self.templates):
            matching = self.match_template(template[0], context2)
            if matching:
                # теперь собираем выходную строку, используя сопоставленные ключевые слова и шаблон
                expanded_phrase = self.generate_output_by_template(template[1], matching, text_utils)
                self.logger.debug('NN_InterpreterNew2 knn-1 generated "%s"', expanded_phrase)
                break

        if not expanded_phrase:
            samples = [Sample(context_phrases, short_phrase)]
            X_data = self.vectorize_samples(samples, text_utils)

            with self.graph.as_default():
                y_pred = self.model.predict(x=X_data, verbose=0)

            y_pred = np.argmax(y_pred[0], axis=-1)
            tokens = [self.index2token[itok] for itok in y_pred]
            expanded_phrase = ''.join(tokens).replace('▁', ' ').strip()
            self.logger.debug('NN_InterpreterNew2 seq2seq generated "%s"', expanded_phrase)
        return expanded_phrase

# -*- coding: utf-8 -*-
"""
Реализация новой модели интерпретации реплик собеседника,
в том числе заполнение пропусков (гэппинг, эллипсис), нормализация грамматического лица.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.

07-06-2020 Полная переделка на новую модель интерпретации (seq2seq with attention)
20-06-2020 Добавка шаблонной модели knn-1
28-07-2020 Исправление ошибки с потерей tf-сессии
04-12-2020 Переделка на seq2seq модель, работающую с новой версией tensorflow
"""

import os
import json
import numpy as np
import logging
import random
import pickle
import itertools

import tensorflow as tf
import sentencepiece as spm

from ruchatbot.layers.seq2seq_model import Seq2SeqEncoder, Seq2SeqDecoder, EOS_TOKEN, BOS_TOKEN
from ruchatbot.bot.base_utterance_interpreter2 import BaseUtteranceInterpreter2


class Sample:
    def __init__(self):
        self.left_str = None
        self.left_tokens = None
        self.right_str = None
        self.right_tokens = None


class NN_InterpreterNew6(BaseUtteranceInterpreter2):
    def __init__(self):
        super(NN_InterpreterNew6, self).__init__()
        self.logger = logging.getLogger('NN_InterpreterNew6')
        self.encoder = None
        self.decoder = None
        self.model_config = None
        self.bpe_model = None
        self.templates = None
        self.graph = None

    def load(self, models_folder):
        self.logger.info('Loading NN_InterpreterNew6 model files from "%s"...', models_folder)

        self.graph = tf.compat.v1.get_default_graph()

        # Эталонные экземпляры для knn-1 модели
        with open(os.path.join(models_folder, 'interpreter_templates2.bin'), 'rb') as f:
            self.templates = pickle.load(f)

        # Файлы нейросетевой модели интерпретации
        with open(os.path.join(models_folder, 'interpreter6.config'), 'r') as f:
            self.model_config = json.load(f)

        self.model_params = self.model_config['model_params']

        bpe_path = os.path.join(models_folder, self.model_config['model_params']['bpe_model_name'] + '.model')
        self.bpe_model = spm.SentencePieceProcessor()
        rc = self.bpe_model.Load(bpe_path)
        assert(rc is True)

        self.encoder = Seq2SeqEncoder(self.model_config['encoder']["vocab_size"],
                                      self.model_config['encoder']["embedding_dim"],
                                      self.model_config['encoder']["enc_units"],
                                      self.model_config['encoder']["batch_sz"])

        self.decoder = Seq2SeqDecoder(self.model_config['decoder']["vocab_size"],
                                      self.model_config['decoder']["embedding_dim"],
                                      self.model_config['decoder']["dec_units"],
                                      self.model_config['decoder']["batch_sz"])

        self.encoder.load_weights(os.path.join(models_folder, 'interpreter_encoder.weights'))
        self.decoder.load_weights(os.path.join(models_folder, 'interpreter_decoder.weights'))

        super(NN_InterpreterNew6, self).load(models_folder)

    def create_prediction_sample(self, context):
        sample = Sample()
        sample.left_str = context
        sample.left_tokens = [BOS_TOKEN] + self.bpe_model.EncodeAsPieces(sample.left_str) + [EOS_TOKEN]
        sample.right_str = ''
        sample.right_tokens = []
        return sample

    def vectorize_samples(self, samples):
        nb_samples = len(samples)

        token2index = self.model_params['token2index']
        max_left_len = self.model_params['max_left_len']
        max_right_len = self.model_params['max_right_len']

        X = np.zeros((nb_samples, max_left_len), dtype=np.int32)
        y = np.zeros((nb_samples, max_right_len), dtype=np.int32)

        for isample, sample in enumerate(samples):
            for itoken, token in enumerate(sample.left_tokens[:max_left_len]):
                if token in token2index:
                    X[isample, itoken] = token2index[token]

            for itoken, token in enumerate(sample.right_tokens[:max_right_len]):
                if token in token2index:
                    y[isample, itoken] = token2index[token]

        return X, y

    def predict_output(self, context):
        sample = self.create_prediction_sample(context)
        input_tensor, _ = self.vectorize_samples([sample])

        index2token = dict((i, t) for t, i in self.model_params['token2index'].items())
        max_length_inp = self.model_params['max_left_len']
        max_length_targ = self.model_params['max_right_len']

        inputs = input_tensor

        start_token_index = self.model_params['token2index'][BOS_TOKEN]

        result = ''

        units = self.model_params['hidden_dim']
        hidden = [tf.zeros((1, units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([start_token_index], 0)

        for t in range(max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()
            new_token = index2token[predicted_id]

            if new_token == EOS_TOKEN:
                break

            result += new_token

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        pred_right = result.replace('▁', ' ').strip()
        return pred_right

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

        # 26-04-2021 Склонятор иногда выдает неупотребляющиеся формы слов. Чтобы не выдавать
        # такие кривые реплики, отключаем knn-1 в этом случае.
        broken_forms = ['человеки', 'годов']
        if any((w in res_words) for w in broken_forms):
            return None

        return ' '.join(res_words)

    def interpret(self, phrases, text_utils):
        context_phrases = phrases[:-1]
        short_phrase = phrases[-1]

        expanded_phrase = None

        if True:
            # Сначала пробуем knn-1 модель, ищем подходящий шаблон
            context2 = [self.prepare_context_line(s, text_utils) for s in phrases]
            for it, template in enumerate(self.templates):
                matching = self.match_template(template[0], context2)
                if matching:
                    # теперь собираем выходную строку, используя сопоставленные ключевые слова и шаблон
                    expanded_phrase = self.generate_output_by_template(template[1], matching, text_utils)
                    if expanded_phrase:
                        self.logger.debug('NN_InterpreterNew2 knn-1 generated "%s"', expanded_phrase)

                    break

        if not expanded_phrase:
            left_phrases = []
            for s in context_phrases:
                s2 = text_utils.wordize_text(s)
                if s2[-1] not in '.?!':
                    s2 += '.'
                left_phrases.append(s2)

            left_phrases.append(text_utils.wordize_text(short_phrase))
            context_str = ' | '.join(left_phrases)

            #with self.graph.as_default():
            #    expanded_phrase = self.predict_output(context_str)
            expanded_phrase = self.predict_output(context_str)

            self.logger.debug('NN_InterpreterNew6 seq2seq context="%s" output="%s"', context_str, expanded_phrase)

        return expanded_phrase

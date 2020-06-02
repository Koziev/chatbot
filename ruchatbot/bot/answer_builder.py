# -*- coding: utf-8 -*-
"""
Группа моделей, выполняющих генерацию текста ответа при заданных текстах предпосылки
и вопроса.

Для проекта чат-бота https://github.com/Koziev/chatbot

15-05-2019 Добавлена генеративная модель построения ответа ("Вероятностная Машина Хомского")
10-08-2019 Эксперимент с новой генеративной моделью построения ответа вместо набора старых
21-05-2020 Полная переработка генеративной модели на одну seq2seq with attention
"""

import os
import logging
import json

import numpy as np
from keras.models import model_from_json
import sentencepiece as spm

# https://github.com/asmekal/keras-monotonic-attention
from ruchatbot.layers.attention_decoder import AttentionDecoder


class AnswerBuilder(object):
    def __init__(self):
        self.logger = logging.getLogger('AnswerBuilder')
        self.trace_enabled = True
        self.model = None

    def load_models(self, models_folder, text_utils):
        self.models_folder = models_folder

        config_path = os.path.join(models_folder, 'nn_seq2seq_pqa_generator.config')
        with open(config_path, 'r') as f:
            computed_params = json.load(f)

        arch_file = os.path.join(models_folder, os.path.basename(computed_params['arch_path']))
        weights_file = os.path.join(models_folder, os.path.basename(computed_params['weights_path']))

        with open(arch_file, 'r') as f:
            self.model = model_from_json(f.read(), {'AttentionDecoder': AttentionDecoder})

        self.model.load_weights(weights_file)

        # Токенизатор
        self.bpe_model = spm.SentencePieceProcessor()
        rc = self.bpe_model.Load(os.path.join(models_folder, computed_params['bpe_model_name'] + '.model'))

        self.token2index = computed_params['token2index']
        self.index2token = dict((i, t) for t, i in computed_params['token2index'].items())

        max_left_len = computed_params['max_left_len']
        max_right_len = computed_params['max_right_len']
        self.seq_len = max(max_left_len, max_right_len)

    def get_w2v_paths(self):
        return []

    def build_answer_text(self, premise_groups, premise_rels, question, text_utils):
        # Определяем способ генерации ответа
        answers = []
        answer_rels = []

        X = np.zeros((1, self.seq_len), dtype=np.int32)

        question_str = ' '.join(text_utils.tokenize(question))
        if question_str[-1] != '?':
            question_str += ' ?'

        for premises, group_rel in zip(premise_groups, premise_rels):
            # Предпосылки и вопрос объединяем в одну строку.
            left_parts = []
            for premise in premises:
                s = ' '.join(text_utils.tokenize(premise))
                if s[-1] not in '.?!':
                    s = s + ' .'
                left_parts.append(s)

            left_parts.append(question_str)
            left_str = ' '.join(left_parts)
            left_tokens = self.bpe_model.EncodeAsPieces(left_str)
            for itoken, token in enumerate(left_tokens):
                X[0, itoken] = self.token2index[token]

            y_pred = self.model.predict(X, verbose=0)
            y_pred = np.argmax(y_pred[0], axis=-1)
            tokens = [self.index2token[itok] for itok in y_pred]
            answer_str = ''.join(tokens).replace('▁', ' ').strip()

            answers.append(answer_str)

            answer_rel = group_rel
            answer_rels.append(answer_rel)

        return answers, answer_rels

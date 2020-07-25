# -*- coding: utf-8 -*-
"""
Группа моделей, выполняющих генерацию текста ответа при заданных текстах предпосылки
и вопроса.

Для проекта чат-бота https://github.com/Koziev/chatbot

15-05-2019 Добавлена генеративная модель построения ответа ("Вероятностная Машина Хомского")
10-08-2019 Эксперимент с новой генеративной моделью построения ответа вместо набора старых
21-05-2020 Полная переработка генеративной модели на одну seq2seq with attention
27-06-2020 Добавлена вторая экспериментальная модель генерации ответа - шаблонная knn-1
"""

import os
import logging
import json
import pickle

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
        self.answer_templates = None

    def load_models(self, models_folder, text_utils):
        self.models_folder = models_folder

        with open(os.path.join(models_folder, 'answer_templates.dat'), 'rb') as f:
            self.answer_templates = pickle.load(f)

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

    def is_important_token2(self, t):
        pos = t[1].split('|')[0]
        if pos in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'ADP', 'PREP'):
            return True

        lemma = t[2]
        if lemma in ('да', 'нет', 'не', 'ни', 'ага'):
            return True

        return False

    def prepare_context_line(self, line, text_utils):
        tokens = text_utils.lemmatize2(line)
        tokens = [(t[0], t[1].split('|'), t[2]) for t in tokens if self.is_important_token2(t)]
        return tokens

    def match_support_template(self, templates, context, text_utils):
        match1 = dict()
        for template, tokens in zip(templates, context):
            if len(template) != len(tokens):
                return None

            for template_item, token in zip(template, tokens):
                if template_item[1] is not None:
                    if not all((tag in token[1]) for tag in template_item[1]):
                        return None

                loc = template_item[2]
                if template_item[0] == token[0]:
                    # формы слов совпали буквально
                    if loc is not None:
                        match1[loc] = token[2]
                else:
                    sim = text_utils.word_similarity(template_item[0], token[0])
                    if sim >= 0.90:
                        # близкие векторы слов в шаблоне и фразе
                        if loc is not None:
                            match1[loc] = token[2]
                    else:
                        return None

        return match1

    def generate_output_by_template(self, output_template, matching, text_utils):
        res_words = []
        for word, location, tags in output_template:
            if location is None:
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
                    form = lemma

                res_words.append(form)

        return ' '.join(res_words)

    def build_using_knn1(self, premises, question, text_utils):
        if len(premises) == 1:
            # Пробуем сопоставить с опорным сэмплом в knn-1
            premise = premises[0]
            context = [self.prepare_context_line(s, text_utils) for s in (premise, question)]

            for i1, (template1, output_template) in enumerate(self.answer_templates):
                matching = self.match_support_template(template1, context, text_utils)
                if matching:
                    out = self.generate_output_by_template(output_template, matching, text_utils)
                    self.logger.debug('Answer generated by knn-1: "%s"', out)
                    return out, 1.0

        return None, None

    def build_answer_text(self, premise_groups, premise_rels, question, text_utils):
        # Определяем способ генерации ответа
        answers = []
        answer_rels = []

        X = np.zeros((1, self.seq_len), dtype=np.int32)

        question_str = ' '.join(text_utils.tokenize(question))
        if question_str[-1] != '?':
            question_str += ' ?'

        for premises, group_rel in zip(premise_groups, premise_rels):
            # Сначала попробуем точную knn-1 модель
            answer, answer_rel = self.build_using_knn1(premises, question, text_utils)
            if answer:
                answers.append(answer)
                answer_rels.append(answer_rel)
            else:
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
                    X[0, itoken] = self.token2index.get(token, 0)

                y_pred = self.model.predict(X, verbose=0)
                y_pred = np.argmax(y_pred[0], axis=-1)
                tokens = [self.index2token[itok] for itok in y_pred]
                answer_str = ''.join(tokens).replace('▁', ' ').strip()

                answers.append(answer_str)

                answer_rel = group_rel
                answer_rels.append(answer_rel)

        return answers, answer_rels

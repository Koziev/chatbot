# -*- coding: utf-8 -*-
"""
Группа моделей, выполняющих генерацию текста ответа при заданных текстах предпосылки
и вопроса.

Для проекта чат-бота https://github.com/Koziev/chatbot

15-05-2019 Добавлена генеративная модель построения ответа ("Вероятностная Машина Хомского")
10-08-2019 Эксперимент с новой генеративной моделью построения ответа вместо набора старых
21-05-2020 Полная переработка генеративной модели на одну seq2seq with attention
27-06-2020 Добавлена вторая экспериментальная модель генерации ответа - шаблонная knn-1
28-07-2020 Исправление ошибки с потерей tf-сессии
04-12-2020 Переделка на seq2seq модель, работающую с новой версией tensorflow
"""

import os
import logging
import json
import pickle

import numpy as np
import tensorflow as tf
from keras.models import model_from_json
import sentencepiece as spm

from ruchatbot.layers.seq2seq_model import Seq2SeqEncoder, Seq2SeqDecoder, EOS_TOKEN, BOS_TOKEN


class Sample:
    def __init__(self):
        self.left_str = None
        self.left_tokens = None
        self.right_str = None
        self.right_tokens = None


class AnswerBuilder(object):
    def __init__(self):
        self.logger = logging.getLogger('AnswerBuilder')
        self.trace_enabled = True
        self.answer_templates = None
        self.graph = None
        self.encoder = None
        self.decoder = None
        self.model_config = None

    def load_models(self, models_folder, text_utils):
        self.logger.info('Loading AnswerBuilder model files from "%s"', models_folder)

        self.models_folder = models_folder

        self.graph = tf.compat.v1.get_default_graph()  # ??? вроде не работает нормально в tf2

        with open(os.path.join(models_folder, 'answer_templates.dat'), 'rb') as f:
            self.answer_templates = pickle.load(f)

        # config_path = os.path.join(models_folder, 'nn_seq2seq_pqa_generator.config')
        # with open(config_path, 'r') as f:
        #     computed_params = json.load(f)
        #
        # arch_file = os.path.join(models_folder, os.path.basename(computed_params['arch_path']))
        # weights_file = os.path.join(models_folder, os.path.basename(computed_params['weights_path']))
        #
        # with open(arch_file, 'r') as f:
        #     self.model = model_from_json(f.read(), {'AttentionDecoder': AttentionDecoder})
        #
        # self.model.load_weights(weights_file)

        # Токенизатор
        #self.bpe_model = spm.SentencePieceProcessor()
        #rc = self.bpe_model.Load(os.path.join(models_folder, computed_params['bpe_model_name'] + '.model'))

        with open(os.path.join(models_folder, 'answer_generator.config'), 'r') as f:
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

        self.encoder.load_weights(os.path.join(models_folder, 'answer_generator_encoder.weights'))
        self.decoder.load_weights(os.path.join(models_folder, 'answer_generator_decoder.weights'))

        self.token2index = self.model_params['token2index']
        self.index2token = dict((i, t) for t, i in self.model_params['token2index'].items())

        self.max_left_len = self.model_params['max_left_len']
        self.max_right_len = self.model_params['max_right_len']

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

        # 26-04-2021 Склонятор иногда выдает неупотребляющиеся формы слов. Чтобы не выдавать
        # такие кривые реплики, отключаем knn-1 в этом случае.
        broken_forms = ['человеки', 'годов']
        if any((w in res_words) for w in broken_forms):
            return None

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

    def build_answer_text(self, premise_groups, premise_rels, question, text_utils):
        answers = []
        answer_rels = []

        qtx = list(text_utils.tokenize(question))
        if qtx[-1] in ['.', '!']:
            qtx = qtx[:-1]
        if qtx[-1] != '?':
            qtx.append('?')
        question_str = text_utils.normalize_delimiters(' '.join(qtx))

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
                        s = s + '.'
                    left_parts.append(s)

                left_parts.append(question_str)
                left_str = ' '.join(left_parts)

                answer_str = self.predict_output(left_str)

                self.logger.debug('AnswerBuilder seq2seq context="%s" output="%s"', left_str, answer_str)

                answers.append(answer_str)

                answer_rel = group_rel
                answer_rels.append(answer_rel)

        return answers, answer_rels

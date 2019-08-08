# -*- coding: utf-8 -*-
"""
Реализация новой модели интерпретации реплик собеседника,
в том числе заполнение пропусков, нормализация грамматического лица.
Формируется набор команд для генеративной грамматики.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

import os
import json
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

from ruchatbot.bot.base_utterance_interpreter2 import BaseUtteranceInterpreter2
from ruchatbot.utils.padding_utils import PAD_WORD, lpad_wordseq, rpad_wordseq

BEG_TOKEN = '<begin>'
END_TOKEN = '<end>'

class Sample(object):
    def __init__(self, question, question_words, short_answer, short_answer_words):
        self.question = question
        self.short_answer = short_answer
        self.question_words = question_words
        self.short_answer_words = short_answer_words


def lpad_wordseq(words, n):
    """ Слева добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """ Справа добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def pad_wordseq(words, n, padding):
    if padding == 'right':
        return rpad_wordseq(words, n)
    else:
        return lpad_wordseq(words, n)


class NN_InterpreterNew2(BaseUtteranceInterpreter2):
    def __init__(self):
        super(NN_InterpreterNew2, self).__init__()
        self.logger = logging.getLogger('NN_InterpreterNew2')
        self.model = None
        self.model_config = None
        self.model_req = None
        self.model_req_config = None

    def load(self, models_folder):
        self.logger.info('Loading NN_InterpreterNew2 model files')

        # Файлы нейросетевой модели интерпретации
        with open(os.path.join(models_folder, 'nn_interpreter_new1.config'), 'r') as f:
            self.model_config = json.load(f)
            self.index2label = dict(self.model_config['index2label'])
            self.index2term = dict(self.model_config['index2term'])
            self.arch_file = self.model_config['arch_file']
            self.weights_file = self.model_config['weights']
            self.padding = self.model_config['padding']
            self.computed_params = self.model_config.copy()
            self.w2v_filename = os.path.basename(self.model_config['w2v_path'])

        arch_filepath = os.path.join(models_folder, os.path.basename(self.arch_file))
        weights_path = os.path.join(models_folder, os.path.basename(self.weights_file))
        with open(arch_filepath, 'r') as f:
            self.model = model_from_json(f.read(), {'CRF': CRF})
            self.model.load_weights(weights_path)

        super(NN_InterpreterNew2, self).load(models_folder)

    def pad_wordseq(self, words, n):
        if self.padding == 'left':
            return lpad_wordseq(words, n)
        else:
            return rpad_wordseq(words, n)

    def vectorize_samples(self, samples, params, computed_params, embeddings):
        padding = params['padding']
        nb_samples = len(samples)
        max_inputseq_len = computed_params['max_inputseq_len']
        max_outputseq_len = computed_params['max_outputseq_len']
        word_dims = computed_params['word_dims']
        #w2v = computed_params['word2vec']
        #nb_labels = computed_params['nb_labels']
        #label2index = computed_params['label2index']
        #term2index = computed_params['term2index']
        #nb_terms = computed_params['nb_terms']

        if params['arch'] == 'bilstm':
            X1_data = np.zeros((nb_samples, max_inputseq_len, word_dims), dtype=np.float32)
            X2_data = np.zeros((nb_samples, max_inputseq_len, word_dims), dtype=np.float32)
            y_data = None  #np.zeros((nb_samples, nb_labels), dtype=np.bool)

            for isample, sample in enumerate(samples):
                words1 = pad_wordseq(sample.question_words, max_inputseq_len, padding)
                embeddings.vectorize_words(self.w2v_filename, words1, X1_data, isample)

                words2 = pad_wordseq(sample.short_answer_words, max_inputseq_len, padding)
                embeddings.vectorize_words(self.w2v_filename, words2, X2_data, isample)
        elif params['arch'] == 'crf':
            max_len = max(max_inputseq_len, max_outputseq_len) + 2

            X1_data = np.zeros((nb_samples, max_len, word_dims), dtype=np.float32)
            X2_data = np.zeros((nb_samples, max_len, word_dims), dtype=np.float32)
            y_data = None  #np.zeros((nb_samples, max_len, nb_terms), dtype=np.bool)

            for isample, sample in enumerate(samples):
                words1 = pad_wordseq(sample.question_words, max_len, padding)
                embeddings.vectorize_words(self.w2v_filename, words1, X1_data, isample)

                words2 = pad_wordseq(sample.short_answer_words, max_len, padding)
                embeddings.vectorize_words(self.w2v_filename,words2, X2_data, isample)

        return X1_data, X2_data, y_data

    def interpret(self, phrases, text_utils, word_embeddings, generative_grammar):
        assert(len(phrases) == 2)

        question = text_utils.remove_terminators(phrases[0])
        short_answer = text_utils.remove_terminators(phrases[1])
        question_words = text_utils.tokenizer.tokenize(question)
        short_answer_words = text_utils.tokenizer.tokenize(short_answer)

        samples = [Sample(question, question_words, short_answer, short_answer_words)]
        X1_data, X2_data, y_data = self.vectorize_samples(samples, self.model_config, self.computed_params, word_embeddings)

        y_pred = self.model.predict({'input1': X1_data, 'input2': X2_data}, verbose=0)
        if self.model_config['arch'] == 'bilstm':
            label = self.index2label[np.argmax(y_pred[0])]
            terms = label.split()
            #print(u'template={}'.format(label))
        elif self.model_config['arch'] == 'crf':
            terms = np.argmax(y_pred[0], axis=-1)
            terms = [self.index2term[i] for i in terms]
            terms = [t for t in terms if t not in (BEG_TOKEN, END_TOKEN)]
            #print('{}\n\n'.format(u' '.join(terms)))

        # Используем полученный список команд генеративной грамматики в terms
        words_bag = [(w, 1.0) for w in (question_words + short_answer_words)]
        template_str = u' '.join(terms).strip()
        all_generated_phrases = generative_grammar.generate_by_terms(template_str,
                                                                     words_bag,
                                                                     text_utils.known_words,
                                                                     use_assocs=False)
        if len(all_generated_phrases) < 1:
            self.logger.error(u'Could not expand answer using template_str="%s"', template_str)
            return None
        else:
            new_phrase = all_generated_phrases[0]
            self.logger.debug(u'NN_Interpreter template="%s" result="%s" rank=%g',
                              template_str, new_phrase.get_str(), new_phrase.get_rank())
            return new_phrase.get_str()

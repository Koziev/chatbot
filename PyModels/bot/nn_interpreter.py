# -*- coding: utf-8 -*-
"""
Нейросетевая реализация модели интерпретации реплик собеседника.
Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

import os
import json
import numpy as np
import logging
import itertools
import random
import pickle

from keras.models import model_from_json
from base_utterance_interpreter import BaseUtteranceInterpreter
from utils.padding_utils import PAD_WORD, lpad_wordseq, rpad_wordseq


class InterpreterSample:
    def __init__(self, phrases, phrase_words, phrase_lemmas):
        self.phrases = phrases[:]
        self.phrase_words = phrase_words[:]
        self.phrase_lemmas = phrase_lemmas[:]


class InterpreterTrellisNode:
    def __init__(self, lemma, word):
        self.lemma = lemma
        self.word = word
        self.is_start = False
        self.is_end = False
        self.best_p = 1.0
        self.best_prev = None

    @classmethod
    def create_start(cls):
        s = InterpreterTrellisNode(u'<S>', u'<S>')
        s.is_start = True
        return s

    @classmethod
    def create_end(cls):
        s = InterpreterTrellisNode(u'<E>', u'<E>')
        s.is_end = True
        return s


class InterpreterTrellisColumn:
    def __init__(self):
        self.cells = []

    def add_cell(self, cell):
        self.cells.append(cell)


def ngrams(s, n):
    return set(itertools.izip(*[s[i:] for i in range(n)]))


class NN_Interpreter(BaseUtteranceInterpreter):
    def __init__(self):
        super(NN_Interpreter, self).__init__()
        self.logger = logging.getLogger('NN_Interpreter')
        self.model = None
        self.model_config = None
        self.model_req = None
        self.model_req_config = None

    def load(self, models_folder):
        self.logger.info('Loading NN_Interpreter model files')

        # Таблицы для трансляции грамматического лица
        with open(os.path.join(models_folder, 'person_change_dictionary.pickle'), 'r') as f:
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
                                  }


        # Файлы нейросетевой модели интерпретации
        arch_filepath = os.path.join(models_folder, 'nn_interpreter.arch')
        weights_path = os.path.join(models_folder, 'nn_interpreter.weights')
        with open(arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(weights_path)
        self.model = m

        with open(os.path.join(models_folder, 'nn_interpreter.config'), 'r') as f:
            self.model_config = json.load(f)

        self.word_dims = self.model_config['word_dims']
        self.w2v_path = self.model_config['w2v_path']
        self.padding = self.model_config['padding']
        self.max_inputseq_len = self.model_config['max_inputseq_len']
        self.max_outseq_len = self.model_config['max_outseq_len']
        self.lemma2id = self.model_config['lemma2id']
        self.max_nb_inputs = self.model_config['max_nb_inputs']

        self.w2v_filename = os.path.basename(self.w2v_path)
        self.id2lemma = dict((id, lemma) for (lemma, id) in self.lemma2id.items())
        self.nb_lemmas = len(self.lemma2id)

    def pad_wordseq(self, words, n):
        if self.padding == 'left':
            return lpad_wordseq(words, n)
        else:
            return rpad_wordseq(words, n)

    def generate_rows(self, nb_inputs, samples, word_embeddings):
        batch_index = 0

        Xn_batch = []
        Xlemmas_batch = []
        for _ in range(nb_inputs):
            x = np.zeros((1, self.max_inputseq_len, self.word_dims), dtype=np.float32)
            Xn_batch.append(x)
            x = np.zeros((1, self.nb_lemmas), dtype=np.float32)
            Xlemmas_batch.append(x)

        inputs = {}
        for iphrase in range(nb_inputs):
            inputs['input{}'.format(iphrase)] = Xn_batch[iphrase]
            inputs['lemmas{}'.format(iphrase)] = Xlemmas_batch[iphrase]

        for irow, sample in enumerate(samples):
            for iphrase, words in enumerate(sample.phrase_words):
                words = self.pad_wordseq(words, self.max_inputseq_len)
                word_embeddings.vectorize_words(self.w2v_filename, words, Xn_batch[iphrase], irow)

            for iphrase, lemmas in enumerate(sample.phrase_lemmas):
                for lemma in lemmas:
                    if lemma not in self.lemma2id:
                        self.logger.error(u'Missing key "{}" in lemma2id'.format(lemma))
                    else:
                        Xlemmas_batch[iphrase][batch_index, self.lemma2id[lemma]] = 1

            batch_index += 1

        yield inputs

    def require_interpretation(self, phrase, text_utils, word_embeddings):
        pass  # todo

    def interpret(self, phrases, text_utils, word_embeddings):
        assert(0 < len(phrases) <= self.max_nb_inputs)

        phrase_words = [text_utils.tokenizer.tokenize(f) for f in text_utils.remove_terminators(phrases)]
        phrase_lemmas = [text_utils.lemmatize(f) for f in text_utils.remove_terminators(phrases)]

        sample = InterpreterSample(phrases, phrase_words, phrase_lemmas)

        input_2grams = set()
        input_1grams = set()
        for phrase in phrases:
            words = text_utils.tokenizer.tokenize(phrase)
            input_2grams.update(ngrams(words, 2))
            input_1grams.update(words)

        probe_samples = []
        probe_samples.append(sample)

        for x in self.generate_rows(self.max_nb_inputs, probe_samples, word_embeddings):
            y_pred = self.model.predict(x=x, verbose=0)
            y_pred = y_pred[0]
            predicted_lemmas = []
            for lemma_v in y_pred:
                lemma_p = [(i, lemma_v[i]) for i in range(len(lemma_v))]
                lemma_p = sorted(lemma_p, key=lambda z: -z[1])
                lx = []
                for lemma_index, p in lemma_p[:10]:
                    if p > 0.01:
                        lemma = self.id2lemma[lemma_index]
                        if lemma != PAD_WORD:
                            lx.append((lemma, p))

                if len(lx) > 0:
                    predicted_lemmas.append(lx)

            # lemma_index = np.argmax(lemma_v)
                # lemma = self.id2lemma[lemma_index]
                # if lemma != PAD_WORD:
                #    predicted_lemmas.append(lemma)

            # Готовим решетку для Витерби, чтобы выбрать оптимальную цепочку словоформ.
            trellis = []
            start = InterpreterTrellisColumn()
            start.add_cell(InterpreterTrellisNode.create_start())
            trellis.append(start)

            for lemma_p_list in predicted_lemmas:
                column = InterpreterTrellisColumn()
                for lemma, p0 in lemma_p_list:
                    if text_utils.lexicon.has_forms(lemma):
                        for form in text_utils.lexicon.get_lemma_forms(lemma):
                            cell = InterpreterTrellisNode(lemma, form)
                            cell.best_p *= p0
                            if form not in input_1grams:
                                cell.best_p *= 0.1
                            column.add_cell(cell)
                    else:
                        self.logger.warn(u'lemma "{}" not in lemma2forms'.format(lemma))
                        form = lemma
                        cell = InterpreterTrellisNode(lemma, form)
                        cell.best_p *= p0
                        if form not in input_1grams:
                            cell.best_p *= 0.5
                        column.add_cell(cell)

                trellis.append(column)

            end = InterpreterTrellisColumn()
            end.add_cell(InterpreterTrellisNode.create_end())
            trellis.append(end)

            # print('Start trellis scan...')
            # Идем по столбцам решетки.
            for icolumn in range(1, len(trellis)):
                column = trellis[icolumn]
                prev_column = trellis[icolumn - 1]
                for cell in column.cells:
                    best_p = -1.0
                    best_prev = None
                    for prev_cell in prev_column.cells:
                        p12 = cell.best_p * prev_cell.best_p + 1e-6 * random.random()
                        n2 = (prev_cell.word, cell.word)
                        if n2 not in input_2grams:
                            p12 *= 0.5

                        if p12 > best_p:
                            best_p = p12
                            best_prev = prev_cell

                    cell.best_prev = best_prev
                    cell.best_p = best_p

                    # print(u'column={} cell={} best_p={} best_prev={}'.format(icolumn, cell.word, best_p, best_prev.word))

            # print('End trellis scan.')

            # Теперь обратный ход - получаем цепочку словоформ.
            cur_cell = end.cells[0]
            selected_cells = []
            while cur_cell is not None:
                selected_cells.append(cur_cell)
                cur_cell = cur_cell.best_prev

            selected_cells = selected_cells[::-1][1:-1]
            words = [cell.word for cell in selected_cells]

            new_phrase = u' '.join(words)
            self.logger.debug(u'NN_Interpreter result={}'.format(new_phrase))
            # print(u'<<<DEBUG>>> NN_Interpreter result={}'.format(new_phrase))
            return new_phrase

        raise NotImplementedError()

    def normalize_person(self, raw_phrase, text_utils, word_embeddings):
        inwords = text_utils.tokenize(raw_phrase)
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

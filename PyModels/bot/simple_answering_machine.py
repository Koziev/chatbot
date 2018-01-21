# -*- coding: utf-8 -*-

from keras.models import model_from_json
from keras.layers import Embedding
from keras.layers.merge import concatenate, add, multiply
from keras.layers import Lambda
from keras import backend as K
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.layers.core import Activation, RepeatVector, Dense, Masking
from keras.layers.wrappers import Bidirectional
from keras.layers import Input
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
import xgboost
from scipy.sparse import lil_matrix
import json
import os
import pickle
import gensim
import numpy as np
import logging

from base_answering_machine import BaseAnsweringMachine
from simple_dialog_session_factory import SimpleDialogSessionFactory
from word_embeddings import WordEmbeddings
from xgb_relevancy_detector import XGB_RelevancyDetector
from xgb_yes_no_model import XGB_YesNoModel
from nn_model_selector import NN_ModelSelector


class SimpleAnsweringMachine(BaseAnsweringMachine):
    """
    Простой чат-бот на основе набора нейросетевых и прочих моделей.
    """

    def __init__(self, facts_storage, text_utils):
        super(SimpleAnsweringMachine,self).__init__()
        self.facts_storage = facts_storage
        self.trace_enabled = False
        self.session_factory = SimpleDialogSessionFactory(self.facts_storage)
        self.text_utils = text_utils

    def get_model_filepath(self, models_folder, old_filepath):
        _, tail = os.path.split(old_filepath)
        return os.path.join( models_folder,  tail )


    def load_models(self, models_folder):
        self.models_folder = models_folder
        # Общие параметры для сеточных моделей
        with open(os.path.join(models_folder, 'qa_model.config'), 'r') as f:
            model_config = json.load(f)

            self.max_inputseq_len = model_config['max_inputseq_len']
            self.max_outputseq_len = model_config['max_outputseq_len']
            self.w2v_path = model_config['w2v_path']
            self.wordchar2vector_path = self.get_model_filepath( models_folder, model_config['wordchar2vector_path'] )
            self.PAD_WORD = model_config['PAD_WORD']
            self.word_dims = model_config['word_dims']

        # Определение релевантности предпосылки и вопроса на основе XGB модели
        self.relevancy_detector = XGB_RelevancyDetector()
        self.relevancy_detector.load(models_folder)

        # Модель для выбора ответов yes|no на базе XGB
        self.yes_no_model = XGB_YesNoModel()
        self.yes_no_model.load(models_folder)

        # Модель для выбора способа генерации ответа
        self.model_selector = NN_ModelSelector()
        self.model_selector.load(models_folder)

        # нейросетевые модели для выбора способа генерации ответа.
        self.models = dict()
        # for model_label in ['model_selector', 'yes_no', 'word_copy']:
        for model_label in ['word_copy3']:
            arch_filepath = os.path.join(models_folder, 'qa_{}.arch'.format(model_label))
            weights_path = os.path.join(models_folder, 'qa_{}.weights'.format(model_label))
            with open(arch_filepath, 'r') as f:
                m = model_from_json(f.read())

            m.load_weights(weights_path)
            self.models[model_label] = m

        with open(os.path.join(models_folder, 'qa_model.config'), 'r') as f:
            self.qa_model_config = json.load(f)

        # --------------------------------------------------------------------------
        # Классификатор грамматического лица на базе XGB
        config_path = os.path.join(models_folder, 'xgb_person_classifier.config')
        with open(config_path, 'r') as f:
            person_classifier_config = json.load(f)

        self.xgb_person_classifier_shingle_len = person_classifier_config['shingle_len']
        self.xgb_person_classifier_shingle2id = person_classifier_config['shingle2id']
        self.xgb_person_classifier_nb_features = person_classifier_config['nb_features']
        self.xgb_person_classifier = xgboost.Booster()
        self.xgb_person_classifier.load_model(self.get_model_filepath( models_folder, person_classifier_config['model_filename']))

        # ------------------------------------------------------------------------------

        # Нейросетевые модели для манипуляции с грамматическим лицом

        # for model_label in ['person_classifier', 'changeable_word']:
        for model_label in ['changeable_word']:
            arch_filepath = os.path.join(models_folder, 'person_change_{}.arch'.format(model_label))
            weights_path = os.path.join(models_folder, 'person_change_{}.weights'.format(model_label))
            with open(arch_filepath, 'r') as f:
                m = model_from_json(f.read())

            m.load_weights(weights_path)
            self.models[model_label] = m

        with open(os.path.join(models_folder, 'person_change_model.config'), 'r') as f:
            self.person_change_model_config = json.load(f)

        # --------------------------------------------------------------------------

        # Упрощенная модель для работы с грамматическим лицом
        with open(os.path.join(models_folder, 'person_change_dictionary.pickle'), 'r') as f:
            model = pickle.load(f)

        self.w1s = model['word_1s']
        self.w2s = model['word_2s']
        self.person_change_1s_2s = model['person_change_1s_2s']
        self.person_change_2s_1s = model['person_change_2s_1s']

        # Загрузка векторных словарей
        self.word_embeddings = WordEmbeddings()
        self.word_embeddings.load_models(self.w2v_path, self.wordchar2vector_path)

    def vectorize_words(self, words, X_batch, irow):
        self.word_embeddings.vectorize_words(words, X_batch, irow )

    def get_person(self, phrase, tokenizer):
        for word in self.text_utils.tokenize(phrase):
            if word in self.w1s:
                return '1s'
            elif word in self.w2s:
                return '2s'
        return '3'

    def change_person(self, phrase, target_person):
        inwords = self.text_utils.tokenize(phrase)
        outwords = []
        for word in inwords:
            if target_person == '2s' and word in self.w1s:
                outwords.append(self.person_change_1s_2s[word])
            elif target_person == '1s' and word in self.w2s:
                outwords.append(self.person_change_2s_1s[word])
            else:
                outwords.append(word)

        return u' '.join(outwords)

    def unknwon_shingle(self, shingle):
        logging.error(u'Shingle "{}" is unknown'.format(shingle))

    def get_session_factory(self):
        return self.session_factory

    def push_phrase(self, interlocutor, phrase):
        session = self.get_session(interlocutor)

        question = self.text_utils.canonize_text(phrase)
        if question == u'#traceon':
            self.trace_enabled = True
            return
        if question == u'#traceoff':
            self.trace_enabled = False
            return
        if question == u'#facts':
            for fact, person, fact_id in self.facts_storage.enumerate_facts(interlocutor):
                print(u'{}'.format(fact))
            return

        question0 = question

        question_words = self.text_utils.tokenize(question)

        # Может потребоваться смена грамматического лица.
        # Сначала определим грамматическое лицо введенного предложения.
        # Для определения грамматического лица вопроса используем XGB классификатор.
        question_wx = self.text_utils.words2str(question_words)
        shingles = self.text_utils.ngrams(question_wx, self.xgb_person_classifier_shingle_len)
        X_data = lil_matrix((1, self.xgb_person_classifier_nb_features), dtype='bool')
        for shingle in shingles:
            X_data[0, self.xgb_person_classifier_shingle2id[shingle]] = True
        D_data = xgboost.DMatrix(X_data)
        y = self.xgb_person_classifier.predict(D_data)
        person = ['1s', '2s', '3'][ int(y[0]) ]

        if self.trace_enabled:
            logging.debug('detected person={}'.format(person))

        #person = get_person(question, tokenizer)
        if person=='1s':
            question = self.change_person(question, '2s')
        elif person=='2s':
            question = self.change_person(question, '1s')


        if question0[-1]==u'.':
            # Утверждение добавляем как факт в базу знаний
            fact_person = '3'
            if person=='1s':
                fact_person='2s'
            elif person=='2s':
                fact_person='1s'
            fact = question
            if self.trace_enabled:
                print(u'Adding [{}] to knowledge base'.format(fact))
            self.facts_storage.store_new_fact((fact, fact_person, '--from dialogue--'))
            return

        if self.trace_enabled:
            logging.debug(u'Question to process={}'.format(question))

        question_words = self.text_utils.tokenize(question)

        # определяем наиболее релевантную предпосылку
        memory_phrases = list(self.facts_storage.enumerate_facts(interlocutor))
        best_premise, best_rel = self.relevancy_detector.get_most_relevant(question, memory_phrases, self.text_utils, self.word_embeddings)
        if self.trace_enabled:
            logging.debug(u'Best premise is "{}" with relevancy={}'.format(best_premise, best_rel))

        max_wordseq_len2 = int(self.qa_model_config['max_inputseq_len'])
        premise_words = self.text_utils.pad_wordseq(self.text_utils.tokenize(best_premise), max_wordseq_len2)
        question_words = self.text_utils.pad_wordseq(self.text_utils.tokenize(question), max_wordseq_len2)

        # Определяем способ генерации ответа
        model_selector = self.model_selector.select_model(premise_str=best_premise,
                                                          question_str=question,
                                                          text_utils=self.text_utils,
                                                          word_embeddings=self.word_embeddings)
        if self.trace_enabled:
            logging.debug('model_selector={}'.format(model_selector))

        answer = u''

        if model_selector==0:
            # yes/no

            # Модель классификации ответа на базе XGB
            y = self.yes_no_model.calc_yes_no(premise_words, question_words, self.text_utils, self.word_embeddings)
            if y<0.5:
                answer = u'нет'
            else:
                answer = u'да'

        elif model_selector==1:
            # wordcopy #3
            # эта модель имеет 2 классификатора на выходе.
            # первый классификатор выбирает позицию начала цепочки, второй - конца.

            X1_probe = np.zeros((1, max_wordseq_len2, self.word_dims), dtype=np.float32)
            X2_probe = np.zeros((1, max_wordseq_len2, self.word_dims), dtype=np.float32)
            self.vectorize_words(premise_words, X1_probe, 0)
            self.vectorize_words(question_words, X2_probe, 0)

            premise_words = self.text_utils.rpad_wordseq(self.text_utils.tokenize(best_premise), max_wordseq_len2)
            question_words = self.text_utils.rpad_wordseq(self.text_utils.tokenize(question), max_wordseq_len2)

            X1_probe.fill(0)
            X2_probe.fill(0)

            self.vectorize_words(premise_words, X1_probe, 0)
            self.vectorize_words(question_words, X2_probe, 0)

            (y1_probe, y2_probe) = self.models['word_copy3'].predict({'input_words1': X1_probe, 'input_words2': X2_probe})
            beg_pos = np.argmax(y1_probe[0])
            end_pos = np.argmax(y2_probe[0])
            words = premise_words[beg_pos:end_pos+1]
            answer = u' '.join(words)

        else:
            answer = 'ERROR: answering model for {} is not implemented'.format(model_selector)

        session.add_to_buffer(answer)

    def pop_phrase(self, interlocutor):
        session = self.get_session(interlocutor)
        return session.extract_from_buffer()


    def get_session(self, interlocutor):
        return self.session_factory[interlocutor]

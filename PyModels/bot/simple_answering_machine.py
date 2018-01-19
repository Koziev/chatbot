# -*- coding: utf-8 -*-

from base_answering_machine import BaseAnsweringMachine
from simple_dialog_session_factory import SimpleDialogSessionFactory

from keras.layers import Embedding
from keras.models import model_from_json
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
        with open(os.path.join(models_folder, 'xgb_relevancy.config'), 'r') as f:
            model_config = json.load(f)

        self.xgb_relevancy_shingle2id = model_config['shingle2id']
        self.xgb_relevancy_shingle_len = model_config['shingle_len']
        self.xgb_relevancy_nb_features = model_config['nb_features']
        self.xgb_relevancy = xgboost.Booster()
        self.xgb_relevancy.load_model( self.get_model_filepath( models_folder,  model_config['model_filename'] ) )

        # Модель для выбора ответов yes|no на базе XGB
        with open(os.path.join(models_folder, 'xgb_yes_no.config'), 'r') as f:
            model_config = json.load(f)

        self.xgb_yesno_shingle2id = model_config['shingle2id']
        self.xgb_yesno_shingle_len = model_config['shingle_len']
        self.xgb_yesno_nb_features = model_config['nb_features']
        self.xgb_yesno_feature_names = model_config['feature_names']
        self.xgb_yesno = xgboost.Booster()
        self.xgb_yesno.load_model( self.get_model_filepath( models_folder,  model_config['model_filename'] ) )

        # нейросетевые модели для выбора способа генерации ответа.
        self.models = dict()
        # for model_label in ['model_selector', 'yes_no', 'word_copy']:
        for model_label in ['model_selector', 'word_copy3']:
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
        self.wc2v = gensim.models.KeyedVectors.load_word2vec_format(self.wordchar2vector_path, binary=False)
        self.wc2v_dims = len(self.wc2v.syn0[0])

        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(self.w2v_path, binary=False)
        self.w2v_dims = len(self.w2v.syn0[0])

    def vectorize_words(self, words, X_batch, irow, w2v, wc2v):
        for iword, word in enumerate(words):
            if word in w2v:
                X_batch[irow, iword, :self.w2v_dims] = self.w2v[word]
            if word in wc2v:
                X_batch[irow, iword, self.w2v_dims:] = self.wc2v[word]

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

    def xgb_yesno_vectorize_sample_x(self, X_data, idata, premise_shingles, question_shingles, shingle2id):
        ps = set(premise_shingles)
        qs = set(question_shingles)
        common_shingles = ps & qs
        notmatched_ps = ps - qs
        notmatched_qs = qs - ps

        nb_shingles = len(shingle2id)

        icol = 0
        for shingle in common_shingles:
            if shingle not in shingle2id:
                print(u'Missing shingle {} in yes_no data'.format(shingle))
            X_data[idata, icol + shingle2id[shingle]] = True

        icol += nb_shingles
        for shingle in notmatched_ps:
            X_data[idata, icol + shingle2id[shingle]] = True

        icol += nb_shingles
        for shingle in notmatched_qs:
            X_data[idata, icol + shingle2id[shingle]] = True

    def unknwon_shingle(self, shingle):
        logging.error(u'Shingle "{}" is unknown'.format(shingle))

    def xgb_relevancy_vectorize_sample_x(self, X_data, idata, premise_shingles, question_shingles, shingle2id):
        ps = set(premise_shingles)
        qs = set(question_shingles)
        common_shingles = ps & qs
        notmatched_ps = ps - qs
        notmatched_qs = qs - ps

        nb_shingles = len(shingle2id)

        icol = 0
        for shingle in common_shingles:
            if shingle not in shingle2id:
                self.unknwon_shingle(shingle)
            else:
                X_data[idata, icol + shingle2id[shingle]] = True

        icol += nb_shingles
        for shingle in notmatched_ps:
            if shingle not in shingle2id:
                self.unknwon_shingle(shingle)
            else:
                X_data[idata, icol + shingle2id[shingle]] = True

        icol += nb_shingles
        for shingle in notmatched_qs:
            if shingle not in shingle2id:
                self.unknwon_shingle(shingle)
            else:
                X_data[idata, icol + shingle2id[shingle]] = True

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
            print('detected person={}'.format(person))

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
            print(u'Question to process={}'.format(question))

        question_words = self.text_utils.tokenize(question)

        # определяем наиболее релевантную предпосылку
        # все предпосылки из текущей базы фактов векторизуем в один тензор, чтобы
        # прогнать его через классификатор разом.
        memory_phrases = list(self.facts_storage.enumerate_facts(interlocutor))
        nb_answers = len(memory_phrases)

        # Поиск наиболее релевантной предпосылки с помощью XGB модели
        X_data = lil_matrix((nb_answers, self.xgb_relevancy_nb_features), dtype='bool')

        best_premise = ''
        best_sim = 0.0
        for ipremise, (premise, premise_person, phrase_code) in enumerate(memory_phrases):
            premise_words = self.text_utils.tokenize(premise)
            question_words = self.text_utils.tokenize(question)
            premise_wx = self.text_utils.words2str(premise_words)
            question_wx = self.text_utils.words2str(question_words)

            premise_shingles = set(self.text_utils.ngrams(premise_wx, self.xgb_person_classifier_shingle_len))
            question_shingles = set(self.text_utils.ngrams(question_wx, self.xgb_person_classifier_shingle_len))

            self.xgb_relevancy_vectorize_sample_x(X_data, ipremise, premise_shingles, question_shingles, self.xgb_relevancy_shingle2id)

        D_data = xgboost.DMatrix(X_data)
        y_probe = self.xgb_relevancy.predict(D_data)

        reslist = []
        for ipremise, (premise, premise_person, phrase_code) in enumerate(memory_phrases):
            sim = y_probe[ipremise]
            reslist.append( (premise, sim) )

        reslist = sorted(reslist, key=lambda z: -z[1])

        best_premise = reslist[0][0]

        if self.trace_enabled:
            print(u'Best premise={}'.format(best_premise))


        # Определяем способ генерации ответа
        max_wordseq_len2 = int(self.qa_model_config['max_inputseq_len'])
        X1_probe = np.zeros((1, max_wordseq_len2, self.word_dims), dtype=np.float32)
        X2_probe = np.zeros((1, max_wordseq_len2, self.word_dims), dtype=np.float32)
        premise_words = self.text_utils.pad_wordseq(self.text_utils.tokenize(best_premise), max_wordseq_len2)
        question_words = self.text_utils.pad_wordseq(self.text_utils.tokenize(question), max_wordseq_len2)
        self.vectorize_words(premise_words, X1_probe, 0, self.w2v, self.wc2v)
        self.vectorize_words(question_words, X2_probe, 0, self.w2v, self.wc2v)
        y_probe = self.models['model_selector'].predict({'input_words1': X1_probe, 'input_words2': X2_probe})
        model_selector = np.argmax( y_probe[0] )
        if self.trace_enabled:
            print('model_selector={}'.format(model_selector))

        answer = u''

        if model_selector==0:
            # yes/no

            # Модель классификации ответа на базе XGB
            premise_wx = self.text_utils.words2str(premise_words)
            question_wx = self.text_utils.words2str(question_words)

            premise_shingles = set(self.text_utils.ngrams(premise_wx, self.xgb_person_classifier_shingle_len))
            question_shingles = set(self.text_utils.ngrams(question_wx, self.xgb_person_classifier_shingle_len))

            X_data = lil_matrix((1, self.xgb_yesno_nb_features), dtype='bool')
            self.xgb_yesno_vectorize_sample_x(X_data, 0, premise_shingles, question_shingles, self.xgb_yesno_shingle2id)

            D_data = xgboost.DMatrix(X_data, feature_names=self.xgb_yesno_feature_names)
            y = self.xgb_yesno.predict(D_data)[0]
            if y<0.5:
                answer = u'нет'
            else:
                answer = u'да'

        elif model_selector==1:
            # wordcopy #3
            # эта модель имеет 2 классификатора на выходе.
            # первый классификатор выбирает позицию начала цепочки, второй - конца.
            premise_words = self.text_utils.rpad_wordseq(self.text_utils.tokenize(best_premise), max_wordseq_len2)
            question_words = self.text_utils.rpad_wordseq(self.text_utils.tokenize(question), max_wordseq_len2)

            X1_probe.fill(0)
            X2_probe.fill(0)

            self.vectorize_words(premise_words, X1_probe, 0, self.w2v, self.wc2v)
            self.vectorize_words(question_words, X2_probe, 0, self.w2v, self.wc2v)

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

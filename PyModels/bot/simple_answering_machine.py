# -*- coding: utf-8 -*-

from keras.models import model_from_json
import xgboost
from scipy.sparse import lil_matrix
import json
import os
import pickle
import numpy as np
import logging

from base_answering_machine import BaseAnsweringMachine
from simple_dialog_session_factory import SimpleDialogSessionFactory
from word_embeddings import WordEmbeddings
from xgb_relevancy_detector import XGB_RelevancyDetector
from xgb_yes_no_model import XGB_YesNoModel
from nn_model_selector import NN_ModelSelector
from xgb_person_classifier_model import XGB_PersonClassifierModel
from nn_wordcopy3 import NN_WordCopy3
from nn_person_change import NN_PersonChange


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
        self.logger = logging.getLogger('SimpleAnsweringMachine')

    def get_model_filepath(self, models_folder, old_filepath):
        _, tail = os.path.split(old_filepath)
        return os.path.join( models_folder,  tail )


    def load_models(self, models_folder):
        self.logger.info(u'Loading models from {}'.format(models_folder))
        self.models_folder = models_folder

        # Загружаем общие параметры для сеточных моделей
        with open(os.path.join(models_folder, 'qa_model_selector.config'), 'r') as f:
            model_config = json.load(f)

            self.max_inputseq_len = model_config['max_inputseq_len']
            self.max_outputseq_len = model_config['max_outputseq_len']
            self.w2v_path = model_config['w2v_path']
            self.wordchar2vector_path = self.get_model_filepath( models_folder, model_config['wordchar2vector_path'] )
            self.PAD_WORD = model_config['PAD_WORD']
            self.word_dims = model_config['word_dims']

        self.qa_model_config = model_config

        # TODO: выбор конкретной реализации для каждого типа моделей сделать внутри базового класса
        # через анализ поля 'engine' в конфигурации модели. Для нейросетевых моделей там будет
        # значение 'nn', для градиентного бустинга - 'xgb'. Таким образом, уберем ненужную связность
        # данного класса и конкретных реализации моделей.

        # Определение релевантности предпосылки и вопроса на основе XGB модели
        self.relevancy_detector = XGB_RelevancyDetector()
        self.relevancy_detector.load(models_folder)

        # Модель для выбора ответов yes|no на базе XGB
        self.yes_no_model = XGB_YesNoModel()
        self.yes_no_model.load(models_folder)

        # Модель для выбора способа генерации ответа
        self.model_selector = NN_ModelSelector()
        self.model_selector.load(models_folder)

        # нейросетевые модели для генерации ответа.
        self.word_copy_model = NN_WordCopy3()
        self.word_copy_model.load(models_folder)

        # Классификатор грамматического лица на базе XGB
        self.person_classifier = XGB_PersonClassifierModel()
        self.person_classifier.load(models_folder)

        # Нейросетевая модель для манипуляции с грамматическим лицом
        self.person_changer = NN_PersonChange()
        self.person_changer.load(models_folder)

        # Загрузка векторных словарей
        self.word_embeddings = WordEmbeddings()
        self.word_embeddings.load_models(self.w2v_path, self.wordchar2vector_path)

    def change_person(self, phrase, target_person):
        return self.person_changer.change_person(phrase, target_person, self.text_utils, self.word_embeddings)

    def get_session_factory(self):
        return self.session_factory

    def push_phrase(self, interlocutor, phrase):
        session = self.get_session(interlocutor)

        question = self.text_utils.canonize_text(phrase)
        if question == u'#traceon':
            self.trace_enabled = True
            return
        elif question == u'#traceoff':
            self.trace_enabled = False
            return
        elif question == u'#facts':
            for fact, person, fact_id in self.facts_storage.enumerate_facts(interlocutor):
                print(u'{}'.format(fact))
            return

        question0 = question

        # Может потребоваться смена грамматического лица.
        # Сначала определим грамматическое лицо введенного предложения.
        person = self.person_classifier.detect_person(question, self.text_utils, self.word_embeddings)
        if self.trace_enabled:
            self.logger.debug('detected person={}'.format(person))

        if person=='1s':
            question = self.change_person(question, '2s')
        elif person=='2s':
            question = self.change_person(question, '1s')

        if question0[-1] in [u'.!']:
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
            self.logger.debug(u'Question to process={}'.format(question))

        #question_words = self.text_utils.tokenize(question)

        # определяем наиболее релевантную предпосылку
        memory_phrases = list(self.facts_storage.enumerate_facts(interlocutor))
        best_premise, best_rel = self.relevancy_detector.get_most_relevant(question, memory_phrases, self.text_utils, self.word_embeddings)
        if self.trace_enabled:
            self.logger.debug(u'Best premise is "{}" with relevancy={}'.format(best_premise, best_rel))

        # Определяем способ генерации ответа
        model_selector = self.model_selector.select_model(premise_str=best_premise,
                                                          question_str=question,
                                                          text_utils=self.text_utils,
                                                          word_embeddings=self.word_embeddings)
        if self.trace_enabled:
            self.logger.debug('model_selector={}'.format(model_selector))

        answer = u''

        if model_selector==0:
            # Ответ генерируется через классификацию на 2 варианта yes|no
            y = self.yes_no_model.calc_yes_no(best_premise, question, self.text_utils, self.word_embeddings)
            if y<0.5:
                answer = u'нет'
            else:
                answer = u'да'

        elif model_selector==1:
            # ответ генерируется через копирование слов из предпосылки.
            answer = self.word_copy_model.generate_answer(best_premise, question, self.text_utils, self.word_embeddings)
        else:
            answer = 'ERROR: answering model for {} is not implemented'.format(model_selector)

        session.add_to_buffer(answer)

    def pop_phrase(self, interlocutor):
        session = self.get_session(interlocutor)
        return session.extract_from_buffer()

    def get_session(self, interlocutor):
        return self.session_factory[interlocutor]

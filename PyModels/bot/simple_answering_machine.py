# -*- coding: utf-8 -*-

import json
import os
import logging
import numpy as np

from base_answering_machine import BaseAnsweringMachine
from simple_dialog_session_factory import SimpleDialogSessionFactory
from word_embeddings import WordEmbeddings
#from xgb_relevancy_detector import XGB_RelevancyDetector
from lgb_relevancy_detector import LGB_RelevancyDetector
from xgb_person_classifier_model import XGB_PersonClassifierModel
from nn_person_change import NN_PersonChange
from answer_builder import AnswerBuilder
from interpreted_phrase import InterpretedPhrase


class SimpleAnsweringMachine(BaseAnsweringMachine):
    """
    Чат-бот на основе набора нейросетевых и прочих моделей (https://github.com/Koziev/chatbot).
    """

    def __init__(self, facts_storage, text_utils):
        super(SimpleAnsweringMachine,self).__init__()
        self.facts_storage = facts_storage
        self.trace_enabled = False
        self.session_factory = SimpleDialogSessionFactory(self.facts_storage)
        self.text_utils = text_utils
        self.logger = logging.getLogger('SimpleAnsweringMachine')
        self.scripting = None

        # Если релевантность факта к вопросу в БФ ниже этого порога, то факт не подойдет
        # для генерации ответа на основе факта.
        self.min_premise_relevancy = 0.2

    def get_model_filepath(self, models_folder, old_filepath):
        """
        Для внутреннего использования - корректирует абсолютный путь
        к файлам данных модели так, чтобы был указанный каталог.
        """
        _, tail = os.path.split(old_filepath)
        return os.path.join( models_folder, tail)

    def load_models(self, models_folder, w2v_folder):
        self.logger.info(u'Loading models from {}'.format(models_folder))
        self.models_folder = models_folder

        # Загружаем общие параметры для сеточных моделей
        with open(os.path.join(models_folder, 'qa_model_selector.config'), 'r') as f:
            model_config = json.load(f)

            self.max_inputseq_len = model_config['max_inputseq_len']
            #self.max_outputseq_len = model_config['max_outputseq_len']
            #self.w2v_path = model_config['w2v_path']
            self.wordchar2vector_path = self.get_model_filepath( models_folder, model_config['wordchar2vector_path'] )
            self.PAD_WORD = model_config['PAD_WORD']
            self.word_dims = model_config['word_dims']

        self.qa_model_config = model_config

        # TODO: выбор конкретной реализации для каждого типа моделей сделать внутри базового класса
        # через анализ поля 'engine' в конфигурации модели. Для нейросетевых моделей там будет
        # значение 'nn', для градиентного бустинга - 'xgb'. Таким образом, уберем ненужную связность
        # данного класса и конкретных реализации моделей.

        # Определение релевантности предпосылки и вопроса на основе XGB модели
        #self.relevancy_detector = XGB_RelevancyDetector()
        self.relevancy_detector = LGB_RelevancyDetector()
        self.relevancy_detector.load(models_folder)

        # Комплексная модель (группа моделей) для генерации текста ответа
        self.answer_builder = AnswerBuilder()
        self.answer_builder.load_models(models_folder)

        # Классификатор грамматического лица на базе XGB
        self.person_classifier = XGB_PersonClassifierModel()
        self.person_classifier.load(models_folder)

        # Нейросетевая модель для манипуляции с грамматическим лицом
        self.person_changer = NN_PersonChange()
        self.person_changer.load(models_folder)

        # Загрузка векторных словарей
        self.word_embeddings = WordEmbeddings()
        self.word_embeddings.load_wc2v_model(self.wordchar2vector_path)
        for p in self.answer_builder.get_w2v_paths():
            p = os.path.join(w2v_folder, os.path.basename(p))
            self.word_embeddings.load_w2v_model(p)

    def set_scripting(self, scripting):
        self.scripting = scripting

    def start_conversation(self, interlocutor):
        """
        Начало общения бота с interlocutor. Ни одной реплики еще не было.
        Бот может поприветствовать собеседника или напомнить ему что-то, если
        в сессии с ним была какая-то напоминалка, т.д. Фразу, которую надо показать собеседнику,
        поместим в буфер выходных фраз с помощью метода say, а внешний цикл обработки уже извлечет ее оттуда
        и напечатает в консоли и т.д.

        :param interlocutor: строковый идентификатор собеседника.
        :return: строка реплики, которую скажет бот.
        """
        session = self.get_session(interlocutor)
        if self.scripting is not None:
            phrase = self.scripting.start_conversation(self, session)
            if phrase is not None:
                self.say(session, phrase)

    def change_person(self, phrase, target_person):
        return self.person_changer.change_person(phrase, target_person, self.text_utils, self.word_embeddings)

    def get_session_factory(self):
        return self.session_factory

    def is_question(self, phrase):
        return phrase[-1] == u'?'

    def interpret_phrase(self, session, raw_phrase):
        interpreted = InterpretedPhrase(raw_phrase)
        phrase = raw_phrase

        # история фраз доступна в session как conversation_history
        # ...

        # TODO: в будущем смена грамматического лица должна уйти в более общую
        # модель интерпретации реплик.
        # ---------------------------------------------------------------------
        # Может потребоваться смена грамматического лица.
        # Сначала определим грамматическое лицо введенного предложения.
        person = self.person_classifier.detect_person(raw_phrase, self.text_utils, self.word_embeddings)
        if self.trace_enabled:
            self.logger.debug('detected person={}'.format(person))

        if person == '1s':
            phrase = self.change_person(raw_phrase, '2s')
        elif person == '2s':
            phrase = self.change_person(raw_phrase, '1s')

        interpreted.interpretation = phrase
        interpreted.is_question = self.is_question(raw_phrase)
        interpreted.phrase_person = person
        return interpreted

    def say(self, session, answer):
        answer_interpretation = InterpretedPhrase(answer)
        answer_interpretation.is_bot_phrase = True
        session.add_to_buffer(answer)
        session.add_phrase_to_history(answer_interpretation)

    def push_phrase(self, interlocutor, phrase):
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

        session = self.get_session(interlocutor)

        # Выполняем интерпретацию фразы с учетом ранее полученных фраз,
        # так что мы можем раскрыть анафору, подставить в явном виде эллипсис и т.д.
        interpreted_phrase = self.interpret_phrase(session, question)

        # Интерпретация фраз и в общем случае реакция на них зависит и от истории
        # общения, поэтому результат интерпретации сразу добавляем в историю.
        session.add_phrase_to_history(interpreted_phrase)

        answer_generated = False

        if not interpreted_phrase.is_question:
            # Утверждение добавляем как факт в базу знаний, в раздел для
            # текущего собеседника.
            # TODO: факты касательно третьих лиц надо вносить в общий раздел базы, а не
            # для текущего собеседника.
            fact_person = '3'
            if interpreted_phrase.phrase_person == '1s':
                fact_person = '2s'
            elif interpreted_phrase.phrase_person == '2s':
                fact_person = '1s'
            fact = interpreted_phrase.interpretation
            if self.trace_enabled:
                print(u'Adding [{}] to knowledge base'.format(fact))
            self.facts_storage.store_new_fact(interlocutor, (fact, fact_person, '--from dialogue--'))

            if self.scripting is not None:
                answer = self.scripting.generate_response4nonquestion(self, interlocutor, interpreted_phrase)
                if answer is not None:
                    answer_generated = True

            if not answer_generated:
                # подбираем подходящую реплику в ответ на не-вопрос собеседника.
                smalltalk_phrases = self.facts_storage.enumerate_smalltalk_replicas()
                best_premise, best_rel = self.relevancy_detector.get_most_relevant(question,
                                                                                   [(item.query, -1, -1) for item in smalltalk_phrases],
                                                                                   self.text_utils,
                                                                                   self.word_embeddings)

                # если релевантность найденной реплики слишком мала, то нужен другой алгоритм...
                for item in smalltalk_phrases:
                    if item.query == best_premise:
                        # выбираем случайный вариант ответа
                        # TODO: уточнить выбор, подбирая наиболее релевантный вариант, так что выдаваемая
                        # реплика будет учитывать либо текущий дискурс, либо ???...
                        # Следует учесть, что ответные реплики в SmalltalkReplicas могут быть ненормализованы,
                        # поэтому их следует сначала нормализовать.
                        answer = np.random.choice(item.answers)
                        answer_generated = True
                        break

            if answer_generated:
                self.say(session, answer)
        else:
            # обрабатываем вопрос
            answer = self.build_answer(interlocutor, interpreted_phrase)
            if answer is not None:
                self.say(session, answer)

            # Возможно, кроме ответа на вопрос, надо выдать еще какую-то реплику.
            # Например, для смены темы разговора.
            if self.scripting is not None:
                additional_speech = self.scripting.generate_after_answer(self, interlocutor, interpreted_phrase, answer)
                if additional_speech is not None:
                    self.say(session, additional_speech)


    def build_answer0(self, interlocutor, interpreted_phrase):
        if self.trace_enabled:
            self.logger.debug(u'Question to process={}'.format(interpreted_phrase.interpretation))

        # определяем наиболее релевантную предпосылку
        memory_phrases = list(self.facts_storage.enumerate_facts(interlocutor))
        best_premise, best_rel = self.relevancy_detector.get_most_relevant(interpreted_phrase.interpretation,
                                                                           memory_phrases,
                                                                           self.text_utils,
                                                                           self.word_embeddings)
        if self.trace_enabled:
            self.logger.debug(u'Best premise is "{}" with relevancy={}'.format(best_premise, best_rel))

        answer = u''
        if best_rel >= self.min_premise_relevancy:
            # генерация ответа на основе выбранной предпосылки.
            answer = self.answer_builder.build_answer_text(best_premise,
                                                           interpreted_phrase.interpretation,
                                                           self.text_utils,
                                                           self.word_embeddings)

        return answer, best_rel


    def build_answer(self, interlocutor, interpreted_phrase):
        answer, answer_confidense = self.build_answer0(interlocutor, interpreted_phrase)
        if answer_confidense < self.min_premise_relevancy:
            # тут нужен алгоритм генерации ответа в условиях, когда
            # у бота нет нужных фактов. Это может быть как ответ "не знаю",
            # так и вариант "нет" для определенных категорий вопросов.
            if self.scripting is not None:
                answer = self.scripting.buid_answer(self, interlocutor, interpreted_phrase)

        return answer

    def pop_phrase(self, interlocutor):
        session = self.get_session(interlocutor)
        return session.extract_from_buffer()

    def get_session(self, interlocutor):
        return self.session_factory[interlocutor]

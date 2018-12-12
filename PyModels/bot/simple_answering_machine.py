# -*- coding: utf-8 -*-

import json
import os
import math
import logging
import numpy as np
import itertools
import operator

from base_answering_machine import BaseAnsweringMachine
from simple_dialog_session_factory import SimpleDialogSessionFactory
from word_embeddings import WordEmbeddings
#from xgb_relevancy_detector import XGB_RelevancyDetector
from lgb_relevancy_detector import LGB_RelevancyDetector
from nn_relevancy_tripleloss import NN_RelevancyTripleLoss
from xgb_person_classifier_model import XGB_PersonClassifierModel
from nn_person_change import NN_PersonChange
from answer_builder import AnswerBuilder
from interpreted_phrase import InterpretedPhrase
from nn_enough_premises_model import NN_EnoughPremisesModel
from nn_synonymy_detector import NN_SynonymyDetector
from nn_synonymy_tripleloss import NN_SynonymyTripleLoss
from jaccard_synonymy_detector import Jaccard_SynonymyDetector
from nn_interpreter import NN_Interpreter
from nn_req_interpretation import NN_ReqInterpretation
from modality_detector import ModalityDetector
from simple_modality_detector import SimpleModalityDetectorRU


class SimpleAnsweringMachine(BaseAnsweringMachine):
    """
    Чат-бот на основе набора нейросетевых и прочих моделей (https://github.com/Koziev/chatbot).
    """

    def __init__(self, text_utils):
        super(SimpleAnsweringMachine,self).__init__()
        self.trace_enabled = False
        self.session_factory = SimpleDialogSessionFactory()
        self.text_utils = text_utils
        self.logger = logging.getLogger('SimpleAnsweringMachine')

        # Если релевантность факта к вопросу в БФ ниже этого порога, то факт не подойдет
        # для генерации ответа на основе факта.
        self.min_premise_relevancy = 0.6

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
        #self.relevancy_detector = LGB_RelevancyDetector()
        self.relevancy_detector = NN_RelevancyTripleLoss()
        self.relevancy_detector.load(models_folder)

        # Модель определения синонимичности двух фраз
        #self.synonymy_detector = NN_SynonymyDetector()
        self.synonymy_detector = NN_SynonymyTripleLoss()
        self.synonymy_detector.load(models_folder)
        #self.synonymy_detector = Jaccard_SynonymyDetector()

        self.interpreter = NN_Interpreter()
        self.interpreter.load(models_folder)

        self.req_interpretation = NN_ReqInterpretation()
        self.req_interpretation.load(models_folder)

        # Определение достаточности набора предпосылок для ответа на вопрос
        self.enough_premises = NN_EnoughPremisesModel()
        self.enough_premises.load(models_folder)

        # Комплексная модель (группа моделей) для генерации текста ответа
        self.answer_builder = AnswerBuilder()
        self.answer_builder.load_models(models_folder)

        # Классификатор грамматического лица на базе XGB
        self.person_classifier = XGB_PersonClassifierModel()
        self.person_classifier.load(models_folder)

        # Нейросетевая модель для манипуляции с грамматическим лицом
        self.person_changer = NN_PersonChange()
        self.person_changer.load(models_folder)

        # Модель определения модальности фраз собеседника
        self.modality_model = SimpleModalityDetectorRU()
        self.modality_model.load(models_folder)

        # Загрузка векторных словарей
        self.word_embeddings = WordEmbeddings()
        self.word_embeddings.load_models(models_folder)
        self.word_embeddings.load_wc2v_model(self.wordchar2vector_path)
        for p in self.answer_builder.get_w2v_paths():
            p = os.path.join(w2v_folder, os.path.basename(p))
            self.word_embeddings.load_w2v_model(p)

        self.word_embeddings.load_w2v_model(self.relevancy_detector.get_w2v_path())

        self.word_embeddings.load_w2v_model(os.path.join(w2v_folder, os.path.basename(self.enough_premises.get_w2v_path())))
        self.logger.debug('All models loaded')

    #def set_scripting(self, scripting):
    #    self.scripting = scripting

    def start_conversation(self, bot, interlocutor):
        """
        Начало общения бота с interlocutor. Ни одной реплики еще не было.
        Бот может поприветствовать собеседника или напомнить ему что-то, если
        в сессии с ним была какая-то напоминалка, т.д. Фразу, которую надо показать собеседнику,
        поместим в буфер выходных фраз с помощью метода say, а внешний цикл обработки уже извлечет ее оттуда
        и напечатает в консоли и т.д.

        :param bot: экземпляр класса BotPersonality
        :param interlocutor: строковый идентификатор собеседника.
        :return: строка реплики, которую скажет бот.
        """
        session = self.get_session(bot, interlocutor)
        if bot.has_scripting():
            phrase = bot.scripting.start_conversation(self, session)
            if phrase is not None:
                self.say(session, phrase)

    def change_person(self, phrase, target_person):
        return self.person_changer.change_person(phrase, target_person, self.text_utils, self.word_embeddings)

    def get_session_factory(self):
        return self.session_factory

    def is_question(self, phrase):
        return self.modality_model.get_modality(phrase, self.text_utils, self.word_embeddings) == ModalityDetector.question

    def translate_order(self, bot, session, raw_phrase):
        order_templates = bot.order_templates.get_templates()
        order2anchor = dict((order, anchor) for (anchor, order) in order_templates)
        phrases = list(order for (anchor, order) in order_templates)
        phrases2 = list((self.text_utils.wordize_text(order),) for (anchor, order) in order_templates)
        canonized2raw = dict((f2[0], f1) for (f1, f2) in itertools.izip(phrases, phrases2))

        best_order, best_sim = self.synonymy_detector.get_most_similar(self.text_utils.wordize_text(raw_phrase),
                                                                       phrases2,
                                                                       self.text_utils,
                                                                       self.word_embeddings,
                                                                       nb_results=1)

        if self.trace_enabled:
            self.logger.info(u'Closest order is "{}" with similarity={}'.format(best_order, best_sim))

        if best_sim > 0.70:
            interpreted_order = order2anchor[canonized2raw[best_order]]
            if self.trace_enabled:
                self.logger.info(u'Phrase "{}" is interpreted as "{}"'.format(raw_phrase, interpreted_order))
            return interpreted_order
        else:
            return None

    def interpret_phrase(self, bot, session, raw_phrase):
        interpreted = InterpretedPhrase(raw_phrase)
        phrase = raw_phrase
        phrase_is_question = self.is_question(phrase)

        # история фраз доступна в session как conversation_history
        person = None

        last_phrase = session.conversation_history[-1] if len(session.conversation_history) > 0 else None
        if len(session.conversation_history) > 0\
            and last_phrase.is_bot_phrase\
            and last_phrase.is_question\
            and not phrase_is_question\
            and self.interpreter is not None:

            if self.req_interpretation.require_interpretation(raw_phrase,
                                                              self.text_utils,
                                                              self.word_embeddings):
                # В отдельной ветке обрабатываем ситуацию, когда бот
                # задал вопрос, на который собеседник дал краткий ответ.
                # с помощью специальной модели мы попробуем восстановить полный
                # текст ответа ообеседника.
                context_phrases = []
                context_phrases.append(last_phrase.interpretation)
                context_phrases.append(raw_phrase)
                phrase = self.interpreter.interpret(context_phrases, self.text_utils, self.word_embeddings)

                # определим грамматическое лицо получившейся интерпретации
                person = self.person_classifier.detect_person(phrase, self.text_utils, self.word_embeddings)

                if person == '2s':  # интерпретация "Тебя зовут Илья" получена из "Меня зовут илья"
                    person = '1s'
                elif person == '1s':
                    person = '2s'

                if self.trace_enabled:
                    self.logger.debug('detected person={}'.format(person))

        if not phrase_is_question and bot.order_templates is not None:
            # попробуем найти шаблон приказа, достаточно похожий на эту фразу
            order_str = self.translate_order(bot, session, phrase)
            if order_str is not None:
                phrase = order_str
                raw_phrase = order_str
                phrase_is_question = self.is_question(phrase)

        if person is None:
            # определим грамматическое лицо введенного предложения.
            person = self.person_classifier.detect_person(raw_phrase, self.text_utils, self.word_embeddings)
            if self.trace_enabled:
                self.logger.debug('detected person={}'.format(person))

            # Может потребоваться смена грамматического лица.
            if person == '1s':
                phrase = self.change_person(raw_phrase, '2s')
            elif person == '2s':
                phrase = self.change_person(raw_phrase, '1s')

        interpreted.interpretation = phrase
        interpreted.is_question = phrase_is_question
        interpreted.phrase_person = person
        return interpreted

    def say(self, session, answer):
        answer_interpretation = InterpretedPhrase(answer)
        answer_interpretation.is_bot_phrase = True
        answer_interpretation.is_question = self.is_question(answer)
        session.add_to_buffer(answer)
        session.add_phrase_to_history(answer_interpretation)

    def does_bot_know_answer(self, question, session):
        """Вернет true, если бот знает ответ на вопрос question"""
        # TODO
        return False

    def calc_discourse_relevance(self, replica, session):
        """Возвращает оценку соответствия реплики replica текущему дискурсу беседы session"""
        # TODO
        return 1.0

    def push_phrase(self, bot, interlocutor, phrase):
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

        session = self.get_session(bot, interlocutor)

        # Выполняем интерпретацию фразы с учетом ранее полученных фраз,
        # так что мы можем раскрыть анафору, подставить в явном виде опущенные составляющие и т.д.,
        # определить, является ли фраза вопросом, фактом или императивным высказыванием.
        interpreted_phrase = self.interpret_phrase(bot, session, question)

        # Интерпретация фраз и в общем случае реакция на них зависит и от истории
        # общения, поэтому результат интерпретации сразу добавляем в историю.
        session.add_phrase_to_history(interpreted_phrase)

        answer_generated = False

        if not interpreted_phrase.is_question:
            # Обработка приказов
            order_processed = False

            if bot.order_templates is not None:
                anchor = bot.order_templates.get_order_anchor(interpreted_phrase.interpretation)
                if anchor is not None:
                    self.process_order(bot, session, interpreted_phrase)
                    order_processed = True

            if not order_processed:
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
                bot.facts.store_new_fact(interlocutor, (fact, fact_person, '--from dialogue--'))

                generated_replicas = []  # список кортежей (подобранная_реплика_бота, вес_реплики)
                if bot.has_scripting():
                    replica = bot.scripting.generate_response4nonquestion(self, interlocutor, interpreted_phrase)
                    if replica is not None:
                        # Не будем допускать, чтобы одна и та же реплика призносилась
                        # ботом более 2х раз
                        if session.count_bot_phrase(replica) <= 2:

                            # Будем учитывать, насколько хорошо предлагаемая реплика ложится в общий
                            # дискурс текущей беседы.
                            discourse_rel = self.calc_discourse_relevance(replica, session)
                            replica_rel = discourse_rel*np.random.rand(0.95, 1.0)

                            # проверить, если answer является репликой-ответом: знает
                            # ли бот ответ на этот вопрос.
                            if self.is_question(replica):
                                if not self.does_bot_know_answer(replica, session):
                                    generated_replicas.add((replica, replica_rel))
                            else:
                                # Добавляется не вопрос.
                                generated_replicas.add((replica, replica_rel))

                if bot.enable_smalltalk:
                    # подбираем подходящую реплику в ответ на не-вопрос собеседника (обычно это
                    # ответ на наш вопрос, заданный ранее).
                    smalltalk_phrases = bot.facts.enumerate_smalltalk_replicas()

                    interlocutor_phrases = session.get_interlocutor_phrases(questions=False, assertions=True)
                    for phrase, timegap in interlocutor_phrases:
                        best_premise, best_rel = self.synonymy_detector.get_most_similar(phrase,
                                                                                         [(item.query, -1, -1) for item in smalltalk_phrases],
                                                                                         self.text_utils,
                                                                                         self.word_embeddings)

                        time_decay = math.exp(-timegap)  # штрафуем фразы, найденные для более старых реплик

                        for item in smalltalk_phrases:
                            if item.query == best_premise:
                                # Следует учесть, что ответные реплики в SmalltalkReplicas могут быть ненормализованы,
                                # поэтому их следует сначала нормализовать.
                                for replica in item.answers:
                                    # Такой вопрос не задавался недавно?
                                    if session.count_bot_phrase(replica) == 0:
                                        # нужно учесть соответствие этой реплики replica текущему дискурсу
                                        # беседы... Например, можно учесть максимальную похожесть на N последних
                                        # реплик...
                                        discourse_rel = self.calc_discourse_relevance(replica, session)

                                        if self.is_question(replica):
                                            # бот не должен задавать вопрос, если он уже знает на него ответ.
                                            if not self.does_bot_know_answer(replica, session):
                                                generated_replicas.append((replica, best_rel*discourse_rel*time_decay, 'debug1'))
                                        else:
                                            generated_replicas.append((replica, best_rel*discourse_rel*time_decay, 'debug2'))
                                break


                    # пробуем найти среди вопросов, которые задавал человек-собеседник недавно,
                    # максимально близкие к вопросам в smalltalk базе.
                    smalltalk_utterances = set()
                    for item in smalltalk_phrases:
                        smalltalk_utterances.update(item.answers)

                    interlocutor_phrases = session.get_interlocutor_phrases(questions=True, assertions=False)
                    for phrase, timegap in interlocutor_phrases:
                        # Ищем ближайшие реплики для данной реплики человека phrase
                        similar_items = self.synonymy_detector.get_most_similar(phrase,
                                                                                [(s, -1, -1) for s in smalltalk_utterances],
                                                                                self.text_utils,
                                                                                self.word_embeddings,
                                                                                nb_results=5
                                                                                )
                        for replica, rel in similar_items:
                            if session.count_bot_phrase(replica) == 0:
                                time_decay = math.exp(-timegap)
                                generated_replicas.append((replica, rel*0.9*time_decay, 'debug3'))

                    # Теперь среди подобранных реплик бота в generated_replicas выбираем
                    # одну, учитывая их вес.
                    if len(generated_replicas) > 0:
                        replica_px = [z[1] for z in generated_replicas]
                        replicas = list(map(operator.itemgetter(0), generated_replicas))
                        sum_p = sum(replica_px) #+1e-7
                        replica_px = [p/sum_p for p in replica_px]
                        answer = np.random.choice(replicas, p=replica_px)
                        answer_generated = True
                    else:
                        answer_generated = False

            if answer_generated:
                self.say(session, answer)
        else:
            # обрабатываем вопрос
            answers = self.build_answers(bot, interlocutor, interpreted_phrase)
            for answer in answers:
                self.say(session, answer)

            # Возможно, кроме ответа на вопрос, надо выдать еще какую-то реплику.
            # Например, для смены темы разговора.
            if len(answers) > 0:
                if bot.has_scripting():
                    additional_speech = bot.scripting.generate_after_answer(bot, self, interlocutor, interpreted_phrase, answers[-1])
                    if additional_speech is not None:
                        self.say(session, additional_speech)


    def process_order(self, bot, session, interpreted_phrase):
        self.logger.info(u'Process order \"{}\"'.format(interpreted_phrase.interpretation))
        bot.process_order(session, interpreted_phrase)


    def build_answers0(self, bot, interlocutor, interpreted_phrase):
        if self.trace_enabled:
            self.logger.debug(u'Question to process={}'.format(interpreted_phrase.interpretation))

        # Нужна ли предпосылка, чтобы ответить на вопрос?
        # Используем модель, которая вернет вероятность того, что
        # пустой список предпосылок достаточен.
        p_enough = self.enough_premises.is_enough(premise_str_list=[],
                                                  question_str=interpreted_phrase.interpretation,
                                                  text_utils=self.text_utils,
                                                  word_embeddings=self.word_embeddings)
        if p_enough > 0.5:
            # Единственный ответ можно построить без предпосылки, например для вопроса "Сколько будет 2 плюс 2?"
            answer_rel = p_enough
            answers, answer_rels = self.answer_builder.build_answer_text([u''], [1.0],
                                                           interpreted_phrase.interpretation,
                                                           self.text_utils,
                                                           self.word_embeddings)
            if len(answers) != 1:
                self.logger.debug(u'Exactly 1 answer was expected for question={}, got {}'.format(interpreted_phrase.interpretation, len(answers)))

            return answers, answer_rels

        else:
            # определяем наиболее релевантную предпосылку
            memory_phrases = list(bot.facts.enumerate_facts(interlocutor))

            best_premises, best_rels = self.relevancy_detector.get_most_relevant(interpreted_phrase.interpretation,
                                                                                 memory_phrases,
                                                                                 self.text_utils,
                                                                                 self.word_embeddings,
                                                                                 nb_results=3)
            if self.trace_enabled:
                self.logger.info(u'Best premise is "{}" with relevancy={}'.format(best_premises[0], best_rels[0]))

            if bot.premise_is_answer:
                # В качестве ответа используется весь текст найденной предпосылки.
                answers = [best_premises[0]]
                answer_rels = [best_rels[0]]
            else:
                premises2 = []
                premise_rels2 = []

                # 30.11.2018 будем использовать только 1 предпосылку и генерировать 1 ответ
                if True:
                    premises2 = [best_premises[:1]]
                    premise_rels2 = best_rels[:1]
                else:
                    max_rel = max(best_rels)
                    for premise, rel in itertools.izip(best_premises[:1], best_rels[:1]):
                        if rel >= self.min_premise_relevancy and rel >= 0.4*max_rel:
                            premises2.append([premise])
                            premise_rels2.append(rel)

                # генерация ответа на основе выбранной предпосылки.
                answers, answer_rels = self.answer_builder.build_answer_text(premises2, premise_rels2,
                                                                             interpreted_phrase.interpretation,
                                                                             self.text_utils,
                                                                             self.word_embeddings)

            return answers, answer_rels


    def build_answers(self, bot, interlocutor, interpreted_phrase):
        answers, answer_confidenses = self.build_answers0(bot, interlocutor, interpreted_phrase)
        if len(answer_confidenses) == 0 or max(answer_confidenses) < self.min_premise_relevancy:
            # тут нужен алгоритм генерации ответа в условиях, когда
            # у бота нет нужных фактов. Это может быть как ответ "не знаю",
            # так и вариант "нет" для определенных категорий вопросов.
            if bot.has_scripting():
                answer = bot.scripting.buid_answer(self, interlocutor, interpreted_phrase)
                answers = [answer]

        return answers

    def pop_phrase(self, bot, interlocutor):
        session = self.get_session(bot, interlocutor)
        return session.extract_from_buffer()

    def get_session(self, bot, interlocutor):
        return self.session_factory.get_session(bot, interlocutor)

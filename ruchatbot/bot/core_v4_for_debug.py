"""
Экспериментальная версия диалогового ядра версии 4.
Основная идея - использование конфабулятора для выбора предпосылок.

16.02.2022 Эксперимент - полностью отказываемся от модели req_interpretation, пусть gpt-модель интерпретатора всегда обрабатывает реплики собеседника.
02.03.2022 Эксперимент - начальный сценарий приветствия теперь активизируется специальной командой в формате [...]
04.02.2022 Эксперимент - объединенная генеративная модель вместо отдельных для читчата, интерпретации, конфабуляции
"""

import sys
import logging.handlers
import os
import argparse
import logging.handlers
import random
import traceback
import itertools
import datetime

import terminaltables

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

from flask import Flask, request, Response
from flask import jsonify


from ruchatbot.bot.entailment_model import EntailmentModel
from ruchatbot.bot.base_utterance_interpreter2 import BaseUtteranceInterpreter2
from ruchatbot.bot.nn_interpreter6 import NN_InterpreterNew6

from ruchatbot.bot.bot_scripting import BotScripting
from ruchatbot.bot.text_utils import TextUtils
#from ruchatbot.bot.lgb_req_interpretation import LGB_ReqInterpretation
from ruchatbot.bot.nn_syntax_validator import NN_SyntaxValidator
from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.bot.lgb_synonymy_detector import LGB_SynonymyDetector
from ruchatbot.bot.lgb_relevancy_detector import LGB_RelevancyDetector
from ruchatbot.bot.modality_detector import ModalityDetector
from ruchatbot.bot.simple_modality_detector import SimpleModalityDetectorRU
from ruchatbot.bot.bot_profile import BotProfile
from ruchatbot.bot.profile_facts_reader import ProfileFactsReader
#from ruchatbot.bot.rugpt_base import RugptBase
#from rugpt_pqa import RugptPQA
from ruchatbot.bot.rugpt_chitchat2 import RugptChitchat
#from ruchatbot.bot.answer_builder import AnswerBuilder
from ruchatbot.bot.rugpt_confabulator import RugptConfabulator
from ruchatbot.bot.rugpt_interpreter import RugptInterpreter



class Utterance:
    def __init__(self, who, text, interpretation=None):
        self.who = who
        self.text = text
        self.interpretation = interpretation

    def get_text(self):
        return self.text

    def is_command(self):
        return self.who == 'X'

    def get_interpretation(self):
        return self.interpretation

    def set_interpretation(self, text):
        self.interpretation = text


class DialogHistory:
    def __init__(self, user_id):
        self.user_id = user_id
        self.messages = []
        self.replies_queue = []

    def get_interlocutor(self):
        return self.user_id

    def enqueue_replies(self, replies):
        """Добавляем в очередь реплики для выдачи собеседнику."""
        self.replies_queue.extend(replies)

    def pop_reply(self):
        if len(self.replies_queue) == 0:
            return ''
        else:
            reply = self.replies_queue[0]
            self.replies_queue = self.replies_queue[1:]
            return reply

    def add_human_message(self, text):
        self.messages.append(Utterance('H', text))

    def add_bot_message(self, text, self_interpretation=None):
        self.messages.append(Utterance('B', text, self_interpretation))

    def add_command(self, command_text):
        self.messages.append(Utterance('X', command_text))

    def get_printable(self):
        lines = []
        for m in self.messages:
            lines.append('{}: {}'.format(m.who, m.text))
        return lines

    def get_last_message(self):
        return self.messages[-1]

    def constuct_interpreter_contexts(self):
        contexts = set()

        max_history = 2

        messages2 = [m for m in self.messages if not m.is_command()]

        for n in range(2, max_history+2):
            steps = []
            for i, message in enumerate(messages2):
                msg_text = message.get_interpretation()
                if msg_text is None:
                    msg_text = message.get_text()

                prev_side = messages2[i-1].who if i > 0 else ''
                if prev_side != message.who:
                    steps.append(msg_text)
                else:
                    s = steps[-1]
                    if s[-1] not in '.?!;:':
                        s += '.'

                    steps[-1] = s + ' ' + msg_text

            last_steps = steps[-n:]
            context = ' | '.join(last_steps)
            contexts.add(context)

        return sorted(list(contexts), key=lambda s: -len(s))

    def construct_entailment_context(self):
        steps = []
        for i, message in enumerate(self.messages):
            msg_text = message.get_text()
            prev_side = self.messages[i-1].who if i > 0 else ''
            if prev_side != message.who:
                steps.append(msg_text)
            else:
                s = steps[-1]
                if s[-1] not in '.?!;:':
                    s += '.'

                steps[-1] = s + ' ' + msg_text

        return ' | '.join(steps[-2:])

    def construct_chitchat_context(self, last_utterance_interpretation, last_utterance_labels):
        labels2 = []
        if last_utterance_labels:
            for x in last_utterance_labels:
                if x[-1] not in '.?!':
                    labels2.append(x+'.')
                else:
                    labels2.append(x)

        if labels2:
            last_utterance_labels_txt = '[{}]'.format(' '.join(labels2))
        else:
            last_utterance_labels_txt = None

        steps = []
        for i, message in enumerate(self.messages):
            msg_text = message.get_text()
            if i == len(self.messages)-1:
                if last_utterance_interpretation:
                    msg_text = last_utterance_interpretation
                else:
                    msg_text = msg_text

            prev_side = self.messages[i-1].who if i > 0 else ''
            if prev_side != message.who:
                steps.append(msg_text)
            else:
                s = steps[-1]
                if s[-1] not in '.?!;:':
                    s += '.'

                steps[-1] = s + ' ' + msg_text

        #return steps[-3:]
        if last_utterance_labels_txt:
            return steps + [last_utterance_labels_txt]
        else:
            return steps


    def set_last_message_interpretation(self, interpretation_text):
        self.messages[-1].set_interpretation(interpretation_text)

    def __len__(self):
        return len(self.messages)


def split_message_text(message, text_utils):
    assertions = []
    questions = []

    for clause in text_utils.split_clauses(message):
        if clause.endswith('?'):
            questions.append(clause)
        else:
            assertions.append(clause)

    return assertions, questions


class GeneratedResponse:
    def __init__(self, algo, prev_utterance_interpretation, text, p, confabulated_facts=None):
        self.algo = algo  # текстовое описание, как получен текст ответа
        self.text = text
        self.prev_utterance_interpretation = prev_utterance_interpretation
        self.p = p
        self.confabulated_facts = confabulated_facts

    def get_text(self):
        return self.text

    def __repr__(self):
        return self.get_text()

    def get_proba(self):
        return self.p

    def get_confabulated_facts(self):
        return self.confabulated_facts

    def get_algo(self):
        return self.algo


def format_outputs(outputs):
    sx = []
    for i, s in enumerate(outputs, start=1):
        sx.append('[{}]"{}"'.format(i, s))
    return ' '.join(sx)


class BotCore:
    def __init__(self):
        self.logger = logging.getLogger('BotCore')

    def load(self, models_dir, text_utils):
        self.text_utils = text_utils

        # =============================
        # Грузим модели.
        # =============================
        self.synonymy_detector = LGB_SynonymyDetector()
        self.synonymy_detector.load(models_dir)

        self.relevancy_detector = LGB_RelevancyDetector()
        self.relevancy_detector.load(models_dir)

        # Модель определения модальности фраз собеседника
        self.modality_model = SimpleModalityDetectorRU()
        self.modality_model.load(models_dir)

        self.syntax_validator = NN_SyntaxValidator()
        self.syntax_validator.load(models_dir)

        self.entailment = EntailmentModel()
        self.entailment.load(models_dir)

        self.generative_model = RugptChitchat()
        self.generative_model.load(models_dir)

        self.base_interpreter = BaseUtteranceInterpreter2()
        self.base_interpreter.load(models_dir)

    def print_dialog(self, dialog):
        logging.debug('='*70)
        table = [['turn', 'side', 'message', 'interpretation']]
        for i, message in enumerate(dialog.messages, start=1):
            interp = message.get_interpretation()
            if interp is None:
                interp = ''
            table.append((i, message.who, message.get_text(), interp))

        for x in terminaltables.AsciiTable(table).table.split('\n'):
            logging.debug('%s', x)
        logging.debug('='*70)

    def store_new_fact(self, fact_text, label, dialog, profile, facts):
        # TODO - проверка на непротиворечивость и неповторение
        self.logger.debug('Storing new fact "%s" in bot="%s" database', fact_text, profile.get_id())
        facts.store_new_fact(dialog.get_interlocutor(), (fact_text, 'unknown', label), True)

    def start_greeting_scenario(self, dialog):
        if random.random() < 0.5:
            command = '[приветствие.]'
        else:
            current_hour = datetime.datetime.now().hour
            if current_hour >= 23 or current_hour < 6:
                command = '[приветствие. сейчас ночь.]'
            elif current_hour in [6, 7, 8, 9]:
                command = '[приветствие. сейчас утро.]'
            elif current_hour in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
                command = '[приветствие. сейчас день.]'
            else:
                command = '[приветствие. сейчас вечер.]'

        dialog.add_command(command)
        chitchat_context = dialog.construct_chitchat_context(last_utterance_interpretation=None, last_utterance_labels=None)
        chitchat_outputs = self.generative_model.generate_chitchat(context_replies=chitchat_context, num_return_sequences=1)
        self.logger.debug('Chitchat@313 start greeting scenario: context="%s" outputs="%s"', ' | '.join(chitchat_context), format_outputs(chitchat_outputs))
        greeting_text = chitchat_outputs[0]
        dialog.add_bot_message(greeting_text)
        return greeting_text

    def normalize_person(self, utterance_text):
        return self.base_interpreter.normalize_person(utterance_text, self.text_utils)

    def flip_person(self, utterance_text):
        return self.base_interpreter.flip_person(utterance_text, self.text_utils)

    def process_human_message(self, dialog, profile, facts):
        # Начинаем обработку реплики собеседника
        interlocutor = dialog.get_interlocutor()
        self.logger.info('Start "process_human_message": message="%s" interlocutor="%s" bot="%s"',
                     dialog.get_last_message().get_text(), interlocutor, profile.get_id())
        self.print_dialog(dialog)

        min_nonsense_threshold = 0.50  # мин. значение синтаксической валидности сгенерированной моделями фразы, чтобы использовать ее дальше
        pqa_rel_threshold = 0.80  # порог отсечения нерелевантных предпосылок

        # Здесь будем накапливать варианты ответной реплики с различной служебной информацией
        responses = []  # [GeneratedResponse]

        memory_phrases = list(facts.enumerate_facts(interlocutor))

        phrase_modality, phrase_person, raw_tokens = self.modality_model.get_modality(dialog.get_last_message().get_text(), self.text_utils)
        must_answer_question = False
        if phrase_modality == ModalityDetector.question:
            must_answer_question = True
        elif phrase_person == 2:
            must_answer_question = True

        # 16-02-2022 интерпретация реплики пользователя выполняется всегда, полагаемся на устойчивость генеративной gpt-модели интерпретатора.
        all_interpretations = []
        interpreter_contexts = dialog.constuct_interpreter_contexts()
        for interpreter_context in interpreter_contexts:
            interpretations = self.generative_model.generate_interpretations([z.strip() for z in interpreter_context.split('|')], num_return_sequences=2)
            self.logger.debug('Interpretation@354: context="%s" outputs="%s"', interpreter_context, ' | '.join(interpretations))

            # Оцениваем "разумность" получившихся интерпретаций, чтобы отсеять заведомо поломанные результаты
            for interpretation in interpretations:
                # может получится так, что возникнет 2 одинаковые интерпретации из формально разных контекстов.
                # избегаем добавления дублирующей интерпретации.
                if not any((interpretation == z[0]) for z in all_interpretations):
                    # Отсекаем дефектные тексты.
                    p_valid = self.syntax_validator.is_valid(interpretation, text_utils=self.text_utils)
                    if p_valid > min_nonsense_threshold:
                        all_interpretations.append((interpretation, p_valid))
                    else:
                        self.logger.debug('Nonsense detector@366: text="%s" p=%5.3f', interpretation, p_valid)

        # В целях оптимизации пока оставляем только самую достоверную интерпретацию.
        # Но вообще мы должны попытаться использовать все сделанные варианты интерпретации и
        # потом уже выбрать лучший вариант реплики
        all_interpretations = sorted(all_interpretations, key=lambda z: -z[1])
        all_interpretations = all_interpretations[:1]

        # Кэш: найденные соответствия между конфабулированными предпосылками и реальными фактами в БД.
        mapped_premises = dict()

        for interpretation, p_interp in all_interpretations:
            # Интерпретация может содержать 2 предложения, типа "я люблю фильмы. ты любишь фильмы?"
            # Каждую клаузу пытаемся обработать отдельно.
            assertionx, questionx = split_message_text(interpretation, self.text_utils)

            input_clauses = [(q, 1.0, True) for q in questionx] + [(a, 0.8, False) for a in assertionx]
            for question_text, question_w, use_confabulation in input_clauses:
                # Ветка ответа на вопрос, в том числе выраженный неявно, например "хочу твое имя узнать!"
                confab_premises = []

                self.logger.debug('Question to process@384: "%s"', question_text)
                # Сначала поищем релевантную информацию в базе фактов
                normalized_phrase_1 = self.normalize_person(question_text)
                premises = []
                rels = []
                premises0, rels0 = self.relevancy_detector.get_most_relevant(normalized_phrase_1, memory_phrases, self.text_utils, nb_results=2)
                for premise, premise_rel in zip(premises0, rels0):
                    if premise_rel >= pqa_rel_threshold:
                        # В базе знаний нашелся релевантный факт.
                        premises.append(premise)
                        rels.append(premise_rel)
                        self.logger.debug('KB lookup@395: query="%s" premise="%s" rel=%f', normalized_phrase_1, premise, premise_rel)

                # С помощью каждого найденного факта (предпосылки) будем генерировать варианты ответа, используя режим PQA читчата
                for premise, premise_relevancy in zip(premises, rels):
                    if premise_relevancy >= pqa_rel_threshold:  # Если найденная в БД предпосылка достаточно релевантна вопросу...
                        confab_premises.append(([premise], premise_relevancy*question_w, 'knowledgebase'))

                if len(confab_premises) == 0 and use_confabulation:
                    phrase_modality, phrase_person, raw_tokens = self.modality_model.get_modality(question_text, self.text_utils)
                    if phrase_person != '2':  # не будем выдумывать факты про собеседника!
                        # В базе знаний ничего релевантного не нашлось.
                        # Просим конфабулятор придумать варианты предпосылок.
                        confabul_context = [interpretation]  #[self.flip_person(interpretation)]
                        # TODO - первый запуск делать с num_return_sequences=10, второй с num_return_sequences=100
                        confabulations = self.generative_model.generate_confabulations(context_replies=confabul_context, num_return_sequences=10)
                        self.logger.debug('Confabulation@413: context="%s" outputs="%s"', confabul_context, format_outputs(confabulations))

                        for confab_text in confabulations:
                            if '|' in confab_text:
                                premises = [z.strip() for z in confab_text.split('|')]
                                confab_premises.append((premises, 1.0, 'confabulation'))
                            else:
                                confab_premises.append(([confab_text], 1.0, 'confabulation'))

                processed_chitchat_contexts = set()

                # Ищем сопоставление придуманных фактов на знания в БД.
                for premises, premises_rel, source in confab_premises:
                    premise_facts = []
                    total_proba = 1.0
                    unmapped_confab_facts = []

                    if source == 'knowledgebase':
                        premise_facts = premises
                        total_proba = 1.0
                    else:
                        for confab_premise in premises:
                            if confab_premise in mapped_premises:
                                memory_phrase, rel = mapped_premises[confab_premise]
                                premise_facts.append(memory_phrase)
                            else:
                                memory_phrase, rel = self.synonymy_detector.get_most_similar(confab_premise, memory_phrases, self.text_utils, nb_results=1)
                                if rel > 0.5:
                                    if memory_phrase != confab_premise:
                                        self.logger.debug('Synonymy@442 text1="%s" text2="%s" score=%5.3f', confab_premise, memory_phrase, rel)

                                    total_proba *= rel
                                    if memory_phrase[-1] not in '.?!':
                                        memory_phrase2 = memory_phrase + '.'
                                    else:
                                        memory_phrase2 = memory_phrase

                                    premise_facts.append(memory_phrase2)
                                    mapped_premises[confab_premise] = (memory_phrase2, rel * premises_rel)
                                else:
                                    # Для этого придуманного факта нет подтверждения в БД. Попробуем его использовать,
                                    # и потом в случае успеха генерации ответа внесем этот факт в БД.
                                    unmapped_confab_facts.append(confab_premise)
                                    premise_facts.append(confab_premise)
                                    mapped_premises[confab_premise] = (confab_premise, 0.80 * premises_rel)

                    if len(premise_facts) == len(premises):
                        # Нашли для всех конфабулированных предпосылок соответствия в базе знаний.
                        if total_proba >= 0.3:
                            # Пробуем сгенерировать ответ, опираясь на найденные в базе знаний предпосылки и заданный собеседником вопрос.
                            chitchat_context = dialog.construct_chitchat_context(interpretation, premise_facts)
                            chitchat_context_str = '|'.join(chitchat_context)
                            if chitchat_context_str not in processed_chitchat_contexts:
                                processed_chitchat_contexts.add(chitchat_context_str)
                                chitchat_outputs = self.generative_model.generate_chitchat(context_replies=chitchat_context, num_return_sequences=5)
                                self.logger.debug('Chitchat@468: context="%s" outputs="%s"', ' | '.join(chitchat_context), format_outputs(chitchat_outputs))
                                for chitchat_output in chitchat_outputs:
                                    # Оценка синтаксической валидности реплики
                                    p_valid = self.syntax_validator.is_valid(chitchat_output, text_utils=self.text_utils)
                                    if p_valid < min_nonsense_threshold:
                                        # Игнорируем поломанные тексты
                                        self.logger.debug('Nonsense detector@474: text="%s" p=%5.3f', chitchat_output, p_valid)
                                        continue

                                    # Оцениваем, насколько этот результат вписывается в контекст диалога
                                    chitchat_context0 = dialog.construct_chitchat_context(last_utterance_interpretation=None, last_utterance_labels=None)
                                    p_entail = self.entailment.predict1(' | '.join(chitchat_context0), chitchat_output)
                                    self.logger.debug('Entailment@457: context="%s" output="%s" p=%5.3f', ' | '.join(chitchat_context0), chitchat_output, p_entail)

                                    p_total = p_valid * p_entail * total_proba
                                    self.logger.debug('Chitchat response scoring@460: context="%s" response="%s" p_valid=%5.3f p_entail=%5.3f p_total=%5.3f',
                                        ' | '.join(chitchat_context), chitchat_output, p_valid, p_entail, p_total)
                                    responses.append(GeneratedResponse('pqa_response',
                                                                       prev_utterance_interpretation=interpretation,
                                                                       text=chitchat_output,
                                                                       p=p_interp * p_total,
                                                                       confabulated_facts=unmapped_confab_facts))

            if len(questionx) == 0:
                message_labels = []

                # Генеративный читчат делает свою основную работу - генерирует ответную реплику.
                chitchat_context = dialog.construct_chitchat_context(interpretation, message_labels)
                chitchat_context0 = dialog.construct_chitchat_context(last_utterance_interpretation=None, last_utterance_labels=None)

                chitchat_outputs = self.generative_model.generate_chitchat(context_replies=chitchat_context, num_return_sequences=5)
                self.logger.debug('Chitchat@499: context="%s" outputs="%s"', ' | '.join(chitchat_context), format_outputs(chitchat_outputs))
                for chitchat_output in chitchat_outputs:
                    # Оценка синтаксической валидности реплики
                    p_valid = self.syntax_validator.is_valid(chitchat_output, text_utils=self.text_utils)
                    if p_valid < min_nonsense_threshold:
                        self.logger.debug('Nonsense detector@504: text="%s" p=%5.3f', chitchat_output, p_valid)
                        continue

                    # Оцениваем, насколько этот результат вписывается в контекст диалога
                    p_entail = self.entailment.predict1(' | '.join(chitchat_context0), chitchat_output)
                    self.logger.debug('Entailment@509: context="%s" output="%s" p=%5.3f', ' | '.join(chitchat_context0), chitchat_output, p_entail)

                    p_total = p_valid * p_entail
                    self.logger.debug('Chitchat response scoring@512: context="%s" response="%s" p_valid=%5.3f p_entail=%5.3f p_total=%5.3f',
                        ' | '.join(chitchat_context), chitchat_output, p_valid, p_entail, p_total)
                    responses.append(GeneratedResponse('chitchat_response',
                                                       prev_utterance_interpretation=interpretation,
                                                       text=chitchat_output,
                                                       p=p_interp * p_total,
                                                       confabulated_facts=None))

        # ===================================================
        # Генерация вариантов ответной реплики закончена.
        # ===================================================

        # Сортируем по убыванию скора
        responses = sorted(responses, key=lambda z: -z.get_proba())

        self.logger.debug('%d responses generated for input_message="%s" interlocutor="%s" bot="%s":', len(responses), dialog.get_last_message().get_text(), interlocutor, profile.get_id())
        table = [['i', 'text', 'score', 'source']]
        for i, r in enumerate(responses, start=1):
            table.append((str(i), r.get_text(), '{:5.3f}'.format(r.get_proba()), r.get_algo()))

        for x in terminaltables.AsciiTable(table).table.split('\n'):
            logging.debug('%s', x)

        if len(responses) == 0:
            self.logger.error('No response generated in context: message="%s" interlocutor="%s" bot="%s"',
                             dialog.get_last_message().get_text(), interlocutor, profile.get_id())
            self.print_dialog(dialog)
            return []

        # Выбираем лучший response, запоминаем интерпретацию последней фразы в истории диалога.
        # 16.02.2022 Идем по списку сгенерированных реплик, проверяем реплику на отсутствие противоречий или заеданий.
        # Если реплика плохая - отбрасываем и берем следующую в сортированном списке.
        self_interpretation = None
        for best_response in responses:
            memory_phrases2 = list(memory_phrases)
            # Если входная обрабатываемая реплика содержит какой-то факт, то его надо учитывать сейчас при поиске
            # релевантных предпосылок. Но так как мы еще не уверены, что именно данный вариант интерпретации входной
            # реплики правильный, то просто соберем временный список с добавленной интерпретацией.
            input_assertions, input_questions = split_message_text(best_response.prev_utterance_interpretation, self.text_utils)
            memory_phrases2 = list(memory_phrases)
            for assertion_text in input_assertions:
                fact_text2 = self.flip_person(assertion_text)
                memory_phrases2.append((fact_text2, '', '(((tmp@554)))'))

            # Вполне может оказаться, что наша ответная реплика - краткая, и мы должны попытаться восстановить
            # полную реплику перед семантическими и прагматическими проверками.
            prevm = best_response.prev_utterance_interpretation # dialog.get_last_message().get_interpretation()
            if prevm is None:
                prevm = dialog.get_last_message().get_text()
            interpreter_context = prevm + ' | ' + best_response.get_text()
            self_interpretation = self.generative_model.generate_interpretations([z.strip() for z in interpreter_context.split('|')], num_return_sequences=1)[0]
            self.logger.debug('Self interpretation@563: context="%s" output="%s"', interpreter_context, self_interpretation)

            is_good_reply = True

            self_assertions, self_questions = split_message_text(self_interpretation, self.text_utils)
            for question_text in self_questions:
                # Реплика содержит вопрос. Проверим, что мы ранее не задавали такой вопрос, и что
                # мы не знаем ответ на этот вопрос. Благодаря этому бот не будет спрашивать снова то, что уже
                # спрашивал или что он просто знает.
                self.logger.debug('Question to process@571: "%s"', question_text)
                premise, rel = self.relevancy_detector.get_most_relevant(question_text, memory_phrases2, self.text_utils, nb_results=1)
                self.logger.debug('KB lookup@573: query="%s" premise="%s" rel=%f', question_text, premise, rel)
                if rel >= pqa_rel_threshold:
                    #self.logger.debug('KB lookup@551: query="%s" premise="%s" rel=%f', question_text, premise, rel)
                    # Так как в БД найден релевантный факт, то бот уже знает ответ на этот вопрос, и нет смысла задавать его
                    # собеседнику снова.
                    is_good_reply = False
                    self.logger.debug('Output response "%s" contains a question "%s" with known answer, so skipping it @555', best_response.get_text(), question_text)
                    break

            if not is_good_reply:
                continue

            # проверяем по БД, нет ли противоречий с утвердительной частью.
            # Генерации реплики, сделанные из предпосылки в БД, не будем проверять.
            if best_response.get_algo() != 'pqa_response':
                for assertion_text in self_assertions:
                    # Ищем релевантный факт в БД
                    premise, rel = self.relevancy_detector.get_most_relevant(assertion_text, memory_phrases2, self.text_utils, nb_results=1)
                    if rel >= pqa_rel_threshold:
                        self.logger.debug('KB lookup@593: query="%s" premise="%s" rel=%f', assertion_text, premise, rel)

                        # Формируем запрос на генерацию ответа через gpt читчата...
                        chitchat_context = '[' + premise + '.] ' + assertion_text + '?'
                        chitchat_outputs = self.generative_model.generate_chitchat(context_replies=[chitchat_context], num_return_sequences=5)
                        self.logger.debug('PQA@598: context="%s" outputs="%s"', chitchat_context, format_outputs(chitchat_outputs))
                        for chitchat_output in chitchat_outputs:
                            # Заглушка - ищем отрицательные частицы
                            words = self.text_utils.tokenize(chitchat_output)
                            if any((w.lower() in ['нет', 'не']) for w in words):
                                is_good_reply = False
                                self.logger.debug('Output response "%s" contains assertion "%s" which contradicts the knowledge base', best_response.get_text(), assertion_text)
                                break

                        if not is_good_reply:
                            break

            if is_good_reply:
                break

        # Если для генерации этой ответной реплики использована интерпретация предыдущей реплики собеседника,
        # то надо запомнить эту интерпретацию в истории диалога.
        dialog.set_last_message_interpretation(best_response.prev_utterance_interpretation)
        input_assertions, input_questions = split_message_text(best_response.prev_utterance_interpretation, self.text_utils)

        for assertion_text in input_assertions:
            # Запоминаем сообщенный во входящей реплике собеседниким факт в базе знаний.
            fact_text2 = self.flip_person(assertion_text)
            self.store_new_fact(fact_text2, '(((dialog@621)))', dialog, profile, facts)

        # Если при генерации ответной реплики использован вымышленный факт, то его надо запомнить в базе знаний.
        if best_response.get_confabulated_facts():
            for f in best_response.get_confabulated_facts():
                self.store_new_fact(f, '(((confabulation@626)))', dialog, profile, facts)

        # Добавляем в историю диалога выбранную ответную реплику
        self.logger.debug('Response for input message "%s" from interlocutor="%s": text="%s" self_interpretation="%s" algorithm="%s" score=%5.3f', dialog.get_last_message().get_text(),
                          dialog.get_interlocutor(), best_response.get_text(), self_interpretation, best_response.algo, best_response.get_proba())

        responses = [best_response.get_text()]

        smalltalk_reply = None
        if False:  #best_response.algo in ['pqa', 'confabulated-pqa']:
            # smalltalk читчат ...
            # Сначала соберем варианты smalltalk-реплик
            smalltalk_responses = []
            if interpretation:
                # Пробуем использовать только интерпретацию в качестве контекста для читчата
                p_context = 0.98  # небольшой штраф за узкий контекст для читчата
                chitchat_outputs = self.chitchat.generate_output(context_replies=[interpretation],
                                                                 num_return_sequences=10)
                # Оставим только вопросы
                chitchat_outputs = [x for x in chitchat_outputs if x.endswith('?')]
                self.logger.debug('Chitchat @466: context="%s" outputs=%s', interpretation,
                                  format_outputs(chitchat_outputs))

                entailment_context = interpretation  # dialog.construct_entailment_context()

                for chitchat_output in chitchat_outputs:
                    # Оценка синтаксической валидности реплики
                    p_valid = self.syntax_validator.is_valid(chitchat_output, text_utils=text_utils)
                    self.logger.debug('Nonsense detector: text="%s" p=%5.3f', chitchat_output, p_valid)

                    # Оцениваем, насколько этот результат вписывается в контекст диалога
                    p_entail = self.entailment.predict1(entailment_context, chitchat_output)
                    self.logger.debug('Entailment @478: context="%s" output="%s" p=%5.3f', entailment_context,
                                      chitchat_output, p_entail)

                    p_total = p_valid * p_entail
                    self.logger.debug(
                        'Chitchat response scoring @481: context="%s" response="%s" p_valid=%5.3f p_entail=%5.3f p_total=%5.3f',
                        entailment_context, chitchat_output, p_valid, p_entail, p_total)
                    smalltalk_responses.append(
                        GeneratedResponse('smalltalk', interpretation, chitchat_output, p_interp * p_context * p_total))

            chitchat_context = dialog.construct_chitchat_context(interpretation)
            if len(chitchat_context) > 1 or chitchat_context[0] != interpretation:
                chitchat_outputs = self.chitchat.generate_output(context_replies=chitchat_context, num_return_sequences=10)
                self.logger.debug('Chitchat @490: context="%s" outputs=%s', ' | '.join(chitchat_context),
                                  format_outputs(chitchat_outputs))
                chitchat_outputs = [x for x in chitchat_outputs if x.endswith('?')]
                for chitchat_output in chitchat_outputs:
                    # Оценка синтаксической валидности реплики
                    p_valid = self.syntax_validator.is_valid(chitchat_output, text_utils=text_utils)
                    self.logger.debug('Nonsense detector: text="%s" p=%5.3f', chitchat_output, p_valid)

                    # Оцениваем, насколько этот результат вписывается в контекст диалога
                    p_entail = self.entailment.predict1(' | '.join(chitchat_context), chitchat_output)
                    self.logger.debug('Entailment @497: context="%s" output="%s" p=%5.3f', ' | '.join(chitchat_context),
                                      chitchat_output, p_entail)

                    p_total = p_valid * p_entail
                    self.logger.debug(
                        'Chitchat response scoring@502: context="%s" response="%s" p_valid=%5.3f p_entail=%5.3f p_total=%5.3f',
                        ' | '.join(chitchat_context), chitchat_output, p_valid, p_entail, p_total)
                    smalltalk_responses.append(GeneratedResponse('smalltalk', interpretation, chitchat_output, p_interp * p_total))

            if smalltalk_responses:
                # Теперь выберем лучшую smalltalk-реплику
                smalltalk_responses = sorted(smalltalk_responses, key=lambda z: -z.get_proba())
                best_smalltalk_response = smalltalk_responses[0]
                self.logger.debug('Best smalltalk response="%s" to user="%s" in bot="%s"',
                                  best_smalltalk_response.get_text(), dialog.get_interlocutor(), profile.get_id())
                smalltalk_reply = best_smalltalk_response.get_text()

        dialog.add_bot_message(best_response.get_text(), self_interpretation)
        if smalltalk_reply:
            dialog.add_bot_message(smalltalk_reply)
            responses.append(smalltalk_reply)

        return responses


def get_user_id(update: Update) -> str:
    user_id = str(update.message.from_user.id)
    return user_id


def tg_start(update, context) -> None:
    user_id = get_user_id(update)
    logging.debug('Entering START callback with user_id=%s', user_id)

    # Старый диалог полностью забываем!
    dialog = DialogHistory(user_id)
    user2dialog[user_id] = dialog

    #msg1 = random.choice(scripting.greetings)
    #dialog.add_bot_message(msg1)
    msg1 = bot.start_greeting_scenario(dialog)
    context.bot.send_message(chat_id=update.message.chat_id, text=msg1)
    logging.debug('Leaving START callback with user_id=%s', user_id)


LIKE = '_Like_'
DISLIKE = '_Dislike_'

last_bot_reply = dict()


def tg_echo(update, context):
    # update.chat.first_name
    # update.chat.last_name
    try:
        user_id = get_user_id(update)

        if user_id in user2dialog:
            dialog = user2dialog[user_id]
        else:
            dialog = DialogHistory(user_id)
            user2dialog[user_id] = dialog

        if update.message.text == LIKE:
            logging.info('LIKE user_id="%s" bot_reply="%s"', user_id, last_bot_reply[user_id])
            return

        if update.message.text == DISLIKE:
            logging.info('DISLIKE user_id="%s" bot_reply="%s"', user_id, last_bot_reply[user_id])
            return

        q = update.message.text
        logging.info('Will reply to "%s" for user="%s" id=%s in chat=%s', q, update.message.from_user.name, user_id, str(update.message.chat_id))

        dialog.add_human_message(q)
        replies = bot.process_human_message(dialog, profile, facts)
        for reply in replies:
            logging.info('Bot reply="%s" to user="%s"', reply, user_id)

        keyboard = [[LIKE, DISLIKE]]
        reply_markup = ReplyKeyboardMarkup(keyboard,
                                           one_time_keyboard=True,
                                           resize_keyboard=True,
                                           per_user=True)

        context.bot.send_message(chat_id=update.message.chat_id, text=reply, reply_markup=reply_markup)
        last_bot_reply[user_id] = reply
    except Exception as ex:
        logging.error(ex)  # sys.exc_info()[0]
        logging.error('Error occured when message "%s" from interlocutor "%s" was being processed: %s', update.message.text, user_id, traceback.format_exc())

# ==================== КОД ВЕБ-СЕРВИСА =======================
flask_app = Flask(__name__)


class SessionFactory(object):
    def __init__(self):
        self.intercolutor2session = dict()

    def get_session(self, interlocutor_id):
        if interlocutor_id not in self.intercolutor2session:
            session = DialogHistory(interlocutor_id)
            self.intercolutor2session[interlocutor_id] = session
            return session
        else:
            return self.intercolutor2session[interlocutor_id]

    def start_conversation(self, interlocutor_id):
        # Забываем предыдущю сессию этого пользователя...
        # TODO - удалять записи в БД.
        session = DialogHistory(interlocutor_id)
        self.intercolutor2session[interlocutor_id] = session
        return session


session_factory = SessionFactory()


@flask_app.route('/start_conversation', methods=["GET"])
def start_conversation():
    user = request.args.get('user', 'anonymous')
    session = session_factory.start_conversation(user)
    msg1 = request.args.get('phrase', random.choice(scripting.greetings))
    logging.debug('start_conversation user="%s" message="%s"', user, msg1)
    session.add_bot_message(msg1)
    session.enqueue_replies([msg1])
    return jsonify({'processed': True})


# response = requests.get(chatbot_url + '/' + 'push_phrase?user={}&phrase={}'.format(user_id, phrase))
@flask_app.route('/push_phrase', methods=["GET"])
def service_push():
    user_id = request.args.get('user', 'anonymous')
    phrase = request.args['phrase']
    logging.debug('push_phrase user="%s" phrase="%s"', user_id, phrase)

    session = session_factory.get_session(user_id)
    session.add_human_message(phrase)

    replies = bot.process_human_message(session, profile, facts)
    session.enqueue_replies(replies)

    response = {'processed': True}
    return jsonify(response)


@flask_app.route('/pop_phrase', methods=["GET"])
def pop_phrase():
    user = request.args.get('user', 'anonymous')
    session = session_factory.get_session(user)
    reply = session.pop_reply()
    return jsonify({'reply': reply})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Сhatbot v4')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--mode', type=str, default='console', choices='console telegram service'.split())
    parser.add_argument('--chatbot_dir', type=str, default='/home/inkoziev/polygon/chatbot')
    parser.add_argument('--log', type=str, default='../../../tmp/core_v4_for_debug.log')
    parser.add_argument('--profile', type=str, default='../../../data/profile_1.json')

    args = parser.parse_args()

    mode = args.mode

    chatbot_dir = args.chatbot_dir
    models_dir = os.path.join(chatbot_dir, 'tmp')
    data_dir = os.path.join(chatbot_dir, 'data')
    tmp_dir = os.path.join(chatbot_dir, 'tmp')
    profile_path = args.profile

    init_trainer_logging(args.log, True)

    # Настроечные параметры аватара собраны в профиле - файле в json формате.
    profile = BotProfile("bot_v4")
    profile.load(profile_path, data_dir, models_dir)

    text_utils = TextUtils()
    text_utils.load_dictionaries(data_dir, models_dir)

    scripting = BotScripting(data_dir)
    scripting.load_rules(profile.rules_path, profile.smalltalk_generative_rules, profile.constants, text_utils)

    # Конкретная реализация хранилища фактов - плоские файлы в utf-8, с минимальным форматированием
    facts = ProfileFactsReader(text_utils=text_utils,
                               profile_path=profile.premises_path,
                               constants=profile.constants)


    bot = BotCore()
    bot.load(models_dir, text_utils)

    if mode == 'debug':
        # Чисто отладочный режим без интерактива.
        # Формируем историю диалога для отладки
        interlocutor = 'test_human'
        dialog = DialogHistory(interlocutor)
        dialog.add_human_message('тебя как зовут?')
        replies = bot.process_human_message(dialog, profile, facts)
        for reply in replies:
            print('B:  {}'.format(reply))
    elif mode == 'telegram':
        # телеграм-бот
        logging.info('Starting telegram bot')
        user2dialog = dict()

        telegram_token = args.token
        if len(telegram_token) == 0:
            telegram_token = input('Enter Telegram token:> ').strip()

        # Телеграм-версия генератора
        tg_bot = telegram.Bot(token=telegram_token).getMe()
        bot_id = tg_bot.name
        logging.info('Telegram bot "%s" id=%s', tg_bot.name, tg_bot.id)

        updater = Updater(token=telegram_token)
        dispatcher = updater.dispatcher

        start_handler = CommandHandler('start', tg_start)
        dispatcher.add_handler(start_handler)

        echo_handler = MessageHandler(Filters.text, tg_echo)
        dispatcher.add_handler(echo_handler)

        logging.getLogger('telegram.bot').setLevel(logging.INFO)
        logging.getLogger('telegram.vendor.ptb_urllib3.urllib3.connectionpool').setLevel(logging.INFO)

        logging.info('Start polling messages for bot %s', tg_bot.name)
        updater.start_polling()
        updater.idle()
    elif mode == 'console':
        # Консольный интерактивный режим для отладки.
        interlocutor = 'test_human'
        dialog = DialogHistory(interlocutor)

        # Начинаем со сценария приветствия.
        bot.start_greeting_scenario(dialog)

        while True:
            for handler in logging.getLogger().handlers:
                handler.flush()
            sys.stdout.flush()

            print('\n'.join(dialog.get_printable()), flush=True)
            h = input('H: ').strip()
            if h:
                dialog.add_human_message(h)
                replies = bot.process_human_message(dialog, profile, facts)
            else:
                break
    elif mode == 'service':
        # Запуск в режиме rest-сервиса
        listen_ip = '127.0.0.1'
        listen_port = '9098'
        logging.info('Going to run flask_app listening %s:%s profile_path=%s', listen_ip, listen_port, profile_path)
        flask_app.run(debug=True, host=listen_ip, port=listen_port)

    print('All done.')

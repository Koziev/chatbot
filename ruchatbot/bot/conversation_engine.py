"""
Экспериментальная версия диалогового ядра версии 5.
Основная идея - использование конфабулятора для выбора предпосылок.

16.02.2022 Эксперимент - полностью отказываемся от модели req_interpretation, пусть gpt-модель интерпретатора всегда обрабатывает реплики собеседника.
02.03.2022 Эксперимент - начальный сценарий приветствия теперь активизируется специальной командой в формате [...]
04.02.2022 Эксперимент - объединенная генеративная модель вместо отдельных для читчата, интерпретации, конфабуляции
11.03.2022 Эксперимент - используем новую модель для pq-релевантность на базе rubert+классификатор
28.03.2022 Эксперимент - переходим на новую модель детектора синонимичности фраз на базе rubert+классификатор
12.09.2022 Возвращаем разделение на модели интерпретации, читчата и конфабуляции
05.10.2022 Эксперимент с отдельной моделью детектирования достаточности контекста для ответа на вопрос (ClosureDetector)
07.10.2022 Переход на пакетную генерацию реплик GPT-моделью
14.10.2022 Закончен переход на пакетную генерацию ответов читчата
15.10.2022 Реализация персистентности фактов в SQLite (класс FactsDatabase)
17.10.2022 Эксперимент с использованием новой модели на базе sentence transformers для подбора фактов БД для вопроса (https://huggingface.co/inkoziev/sbert_pq)
20.10.2022 Переходим на модель оппределения перефразировок на архитектуре sentence transformer.
01.11.2022 Рефакторинг: код для запуска в консоли, телеграм-бота и rest api сервиса вынесен в отдельные модули, см. подкаталог frontend
05.11.2022 Эксперимент с использованием новой модели для раскрытия неполных реплик на базе rut5
"""

from typing import List, Set, Dict, Tuple, Optional
import logging.handlers
import os
import logging.handlers
import random
import datetime
import json

import terminaltables

import torch
import transformers

from ruchatbot.bot.base_utterance_interpreter2 import BaseUtteranceInterpreter2
from ruchatbot.bot.text_utils import TextUtils
from ruchatbot.utils.udpipe_parser import UdpipeParser
from ruchatbot.utils.logging_helpers import init_trainer_logging
from ruchatbot.bot.sbert_paraphrase_detector import SbertSynonymyDetector
from ruchatbot.bot.modality_detector import ModalityDetector
from ruchatbot.bot.simple_modality_detector import SimpleModalityDetectorRU
from ruchatbot.bot.bot_profile import BotProfile
from ruchatbot.bot.profile_facts_reader import ProfileFactsReader
#from ruchatbot.bot.rugpt_interpreter import RugptInterpreter
from ruchatbot.bot.rut5_interpreter import RuT5Interpreter
from ruchatbot.bot.rugpt_confabulator import RugptConfabulator
from ruchatbot.bot.rugpt_chitchat import RugptChitChat
from ruchatbot.bot.sbert_relevancy_detector import SbertRelevancyDetector
from ruchatbot.bot.closure_detector_2 import RubertClosureDetector
from ruchatbot.bot.ruwordnet_relevancy_scorer import RelevancyScorer
from ruchatbot.bot.facts_database import FactsDatabase


class Utterance:
    def __init__(self, who, text, interpretation=None):
        self.who = who
        self.text = text
        self.interpretation = interpretation

    def get_text(self):
        return self.text

    def __repr__(self):
        return '{}: {}'.format(self.who, self.text)

    def is_command(self):
        return self.who == 'X'

    def get_interpretation(self):
        return self.interpretation

    def set_interpretation(self, text):
        self.interpretation = text


class DialogHistory(object):
    """История диалога с одним конкретным собеседником"""

    def __init__(self, user_id):
        self.user_id = user_id
        self.messages = []
        self.replies_queue = []

    def get_interlocutor(self):
        return self.user_id

    def enqueue_replies(self, replies):
        """Добавляем в очередь сгенерированные реплики для выдачи собеседнику."""
        self.replies_queue.extend(replies)

    def pop_reply(self):
        """
        Внешний код с помощью этого метода извлекает из очереди сгенерированных реплик
        очередную реплику для показа ее собеседнику.

        Если в очереди нет реплик, то вернем пустую строку.
        """
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
        self.replies_queue.append(text)

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

    def construct_chitchat_context(self, last_utterance_interpretation, last_utterance_labels, max_depth=10, include_commands=False):
        labels2 = []
        if last_utterance_labels:
            for x in last_utterance_labels:
                x = x[0].upper() + x[1:]
                if x[-1] not in '.?!':
                    labels2.append(x+'.')
                else:
                    labels2.append(x)

        if labels2:
            last_utterance_additional_txt = ' '.join(labels2)
        else:
            last_utterance_additional_txt = None

        steps = []
        for i, message in enumerate(self.messages):
            if not include_commands and message.is_command():
                continue

            msg_text = message.get_text()
            if i == len(self.messages)-1:
                if last_utterance_interpretation:
                    if last_utterance_additional_txt:
                        msg_text = last_utterance_additional_txt + ' ' + last_utterance_interpretation
                    else:
                        msg_text = last_utterance_interpretation
                else:
                    if last_utterance_additional_txt:
                        msg_text = last_utterance_additional_txt + ' ' + msg_text

            prev_side = self.messages[i-1].who if i > 0 else ''
            if prev_side != message.who:
                steps.append(msg_text)
            else:
                s = steps[-1]
                if s[-1] not in '.?!;:':
                    s += '.'

                steps[-1] = s + ' ' + msg_text

        return steps[-max_depth:]

    def set_last_message_interpretation(self, interpretation_text):
        self.messages[-1].set_interpretation(interpretation_text)

    def __len__(self):
        return len(self.messages)


class ConversationSession(object):
    def __init__(self, interlocutor_id, bot_profile, text_utils, facts_db):
        self.interlocutor_id = interlocutor_id
        self.bot_profile = bot_profile
        self.dialog = DialogHistory(interlocutor_id)
        self.facts = ProfileFactsReader(text_utils=text_utils,
                                        profile_path=bot_profile.premises_path,
                                        constants=bot_profile.constants,
                                        facts_db=facts_db)
    def pop_reply(self):
        return self.dialog.pop_reply()

    def enqueue_replies(self, replies):
        self.dialog.enqueue_replies(replies)


class SessionFactory(object):
    def __init__(self, bot_profile, text_utils, facts_db):
        self.bot_profile = bot_profile
        self.text_utils = text_utils
        self.interlocutor2session = dict()
        self.facts_db = facts_db

    def get_session(self, interlocutor_id):
        if interlocutor_id not in self.interlocutor2session:
            return self.start_conversation(interlocutor_id)
        else:
            return self.interlocutor2session[interlocutor_id]

    def start_conversation(self, interlocutor_id):
        session = ConversationSession(interlocutor_id, self.bot_profile, self.text_utils, self.facts_db)
        self.interlocutor2session[interlocutor_id] = session
        return session


class ResponseGenerationPromise(object):
    def __init__(self):
        pass

    @staticmethod
    def make_explicit(text: str):
        r = ResponseGenerationPromise()
        r.text = text
        r.chitchat_generation_context = None
        return r

    @staticmethod
    def make_promise(chitchat_generation_context: List[str]):
        r = ResponseGenerationPromise()
        r.text = None
        r.chitchat_generation_context = chitchat_generation_context
        return r

    def __repr__(self):
        if self.text is not None:
            return self.text
        else:
            return 'promise⦃' + ' | '.join(self.chitchat_generation_context) + '⦄'

    def is_promised(self) -> bool:
        return self.text is None

    def is_generated(self) -> bool:
        return self.text is not None


class GeneratedResponse(object):
    def __init__(self, algo, generation_promise, prev_utterance_interpretation, p, confabulated_facts=None, context=None):
        self.algo = algo  # текстовое описание, как получен текст ответа
        self.generation_promise = generation_promise  # если генерация ответа отложена на потом или текст ууже сгенерирован
        self.prev_utterance_interpretation = prev_utterance_interpretation
        self.p = p
        self.p_entail = 1.0
        self.confabulated_facts = confabulated_facts
        self.context = context

    def set_p_entail(self, p_entail):
        self.p_entail = p_entail

    def get_text(self):
        return str(self.generation_promise)

    def get_context(self):
        return self.context

    def __repr__(self):
        return self.get_text()

    def get_proba(self):
        return self.p * self.p_entail

    def get_confabulated_facts(self):
        return self.confabulated_facts

    def get_algo(self):
        return self.algo


class BotCore:
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.logger = logging.getLogger('BotCore')
        self.min_nonsense_threshold = 0.50  # мин. значение синтаксической валидности сгенерированной моделями фразы, чтобы использовать ее дальше
        self.pqa_rel_threshold = 0.80  # порог отсечения нерелевантных предпосылок


    def load_bert(self, bert_path):
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
        self.bert_model = transformers.BertModel.from_pretrained(bert_path)
        self.bert_model.to(self.device)
        self.bert_model.eval()

    def load(self, models_dir, text_utils):
        self.text_utils = text_utils

        # =============================
        # Грузим модели.
        # =============================
        self.relevancy_detector = SbertRelevancyDetector(device=self.device)
        self.relevancy_detector.load(os.path.join(models_dir, 'sbert_pq'))

        self.synonymy_detector = SbertSynonymyDetector(device=self.device)
        self.synonymy_detector.load(os.path.join(models_dir, 'sbert_synonymy'))

        with open(os.path.join(models_dir, 'closure_detector_2.cfg'), 'r') as f:
            cfg = json.load(f)
            self.closure_detector = RubertClosureDetector(device=self.device, **cfg)
            self.closure_detector.load_weights(os.path.join(models_dir, 'closure_detector_2.pt'))
            self.closure_detector.bert_model = self.bert_model
            self.closure_detector.bert_tokenizer = self.bert_tokenizer

        # Ой, тут грузим второй экземпляр синт. парсера. Надо потом оптимизировать и оставить один парсер в TextUtils()

        parser = UdpipeParser()
        parser.load(models_dir)

        self.p2q_scorer = RelevancyScorer(parser)
        self.p2q_scorer.load(models_dir)

        # Модель определения модальности фраз собеседника
        self.modality_model = SimpleModalityDetectorRU()
        self.modality_model.load(models_dir)

        #self.syntax_validator = NN_SyntaxValidator()
        #self.syntax_validator.load(models_dir)

        #self.entailment = EntailmentModel(self.device)
        #self.entailment.load(models_dir, self.bert_model, self.bert_tokenizer)

        #self.interpreter = RugptInterpreter()
        self.interpreter = RuT5Interpreter()
        self.interpreter.load(models_dir)

        self.confabulator = RugptConfabulator()
        self.confabulator.load(models_dir)

        self.chitchat = RugptChitChat()
        self.chitchat.load(os.path.join(models_dir, 'rugpt_npqa'))

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
        self.logger.debug('Storing new fact 〚%s〛 in bot="%s" database', fact_text, profile.get_id())
        facts.store_new_fact(dialog.get_interlocutor(), fact_text, label, True)

    def start_greeting_scenario(self, session):
        dialog = session.dialog
        if random.random() < 0.5:
            command = 'Поприветствуй меня!'
        else:
            current_hour = datetime.datetime.now().hour
            if current_hour >= 23 or current_hour < 6:
                command = 'Сейчас ночь. Поприветствуй меня!'
            elif current_hour in [6, 7, 8, 9]:
                command = 'Сейчас утро. Поприветствуй меня!'
            elif current_hour in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
                command = 'Сейчас день. Поприветствуй меня!'
            else:
                command = 'Сейчас вечер. Поприветствуй меня!'

        dialog.add_command(command)
        chitchat_context = dialog.construct_chitchat_context(last_utterance_interpretation=None, last_utterance_labels=None, include_commands=True)
        chitchat_outputs = self.chitchat.generate_chitchat(context_replies=chitchat_context, num_return_sequences=1)
        self.logger.debug('Chitchat@373 start greeting scenario: context=〚%s〛 outputs=%s', ' | '.join(chitchat_context), format_outputs(chitchat_outputs))
        greeting_text = chitchat_outputs[0]
        dialog.add_bot_message(greeting_text)
        return greeting_text

    def normalize_person(self, utterance_text):
        return self.base_interpreter.normalize_person(utterance_text, self.text_utils)

    def flip_person(self, utterance_text):
        return self.base_interpreter.flip_person(utterance_text, self.text_utils)

    def process_human_message(self, session):
        """ Process the message from interlocutor in context of session dialogue """
        dialog = session.dialog
        profile = session.bot_profile
        facts = session.facts

        interlocutor = dialog.get_interlocutor()
        self.logger.info('Start "process_human_message": message=〚%s〛 interlocutor="%s" bot="%s"', dialog.get_last_message().get_text(), interlocutor, profile.get_id())
        self.print_dialog(dialog)

        # Здесь будем накапливать варианты ответной реплики с различной служебной информацией
        responses = []  # [GeneratedResponse]

        # Факты в базе знаний, известные на момент начала обработки этой входной реплики
        memory_phrases = list(facts.enumerate_facts(interlocutor))

        # Сначала попробуем использовать весь текст реплики собесеника в качестве контекста P(0)QA.
        # При этом реплика может содержать много предложений, из которых только одно будет вопросом. Например:
        #
        # "У Маши было 3 конфеты. 2 она съела. Сколько конфет осталось у Маши?"
        #
        # Никакой другой информации для ответа на вопрос при этом не требуется.
        # Небольшое ограничение: входим в эту ветку только в случае, если реплика заканчивается знаком вопроса.
        all_answered_texts = set()
        if dialog.get_last_message().get_text().endswith('?'):
            text0 = dialog.get_last_message().get_text()
            rel_p0q = self.closure_detector.calc_label(text0)
            self.logger.debug('Closure detector P(0)Q @460: text=〚%s〛 rel_p0q=%5.3f', text0, rel_p0q)
            if rel_p0q > self.pqa_rel_threshold:
                p0qa_responses = self.generate_p0qa_reply(dialog=dialog, prev_utterance_interpretation=text0, reply_text=text0, rel_p0q=rel_p0q)
                responses.extend(p0qa_responses)
                all_answered_texts.add(text0)

        # 16-02-2022 интерпретация реплики пользователя выполняется всегда, полагаемся на устойчивость генеративной gpt-модели интерпретатора.
        all_interpretations = []
        interpreter_contexts = dialog.constuct_interpreter_contexts()
        for interpreter_context in interpreter_contexts:
            interpretations = self.interpreter.interpret([z.strip() for z in interpreter_context.split('|')], num_return_sequences=2)
            self.logger.debug('Interpretation@471: context=〚%s〛 outputs=〚%s〛', interpreter_context, format_outputs(interpretations))

            # Оцениваем "разумность" получившихся интерпретаций, чтобы отсеять заведомо поломанные результаты
            for interpretation in interpretations:
                # может получится так, что возникнет 2 одинаковые интерпретации из формально разных контекстов.
                # избегаем добавления дублирующей интерпретации.
                if not any((interpretation == z[0]) for z in all_interpretations):
                    # Отсекаем дефектные тексты.
                    p_valid = 1.0  #self.syntax_validator.is_valid(interpretation, text_utils=self.text_utils)
                    if p_valid > self.min_nonsense_threshold:
                        all_interpretations.append((interpretation, p_valid))
                    else:
                        self.logger.debug('Nonsense detector@483: text=〚%s〛 p=%5.3f', interpretation, p_valid)

        # В целях оптимизации пока оставляем только самую достоверную интерпретацию.
        # Но вообще мы должны попытаться использовать все сделанные варианты интерпретации и
        # потом уже выбрать лучший вариант реплики
        all_interpretations = sorted(all_interpretations, key=lambda z: -z[1])
        all_interpretations = all_interpretations[:1]

        # Кэш: найденные соответствия между конфабулированными предпосылками и реальными фактами в БД.
        mapped_premises = dict()
        for interpretation, p_interp in all_interpretations:
            # Вторая попытка применить p(0)q схему, теперь уже для интерпретации
            rel_p0q = self.closure_detector.calc_label(interpretation)
            self.logger.debug('Closure detector P(0)Q @496: text=〚%s〛 rel_p0q=%5.3f', interpretation, rel_p0q)
            if rel_p0q > self.pqa_rel_threshold:
                p0qa_responses = self.generate_p0qa_reply(dialog=dialog, prev_utterance_interpretation=interpretation, reply_text=interpretation, rel_p0q=rel_p0q * p_interp)
                responses.extend(p0qa_responses)
                all_answered_texts.add(interpretation)

            # Интерпретация может содержать 2 предложения, типа "Я люблю фильмы. Ты любишь фильмы?"
            # Каждую клаузу пытаемся обработать отдельно.
            assertionx, questionx = split_message_text(interpretation, self.text_utils)

            input_clauses = [(q, 1.0, True) for q in questionx] + [(a, 0.8, False) for a in assertionx]
            for question_text, question_w, use_confabulation in input_clauses:
                # Ветка ответа на вопрос, в том числе выраженный неявно, например "хочу твое имя узнать!"

                if question_text in all_answered_texts:
                    continue

                all_answered_texts.add(question_text)

                confab_premises = []

                self.logger.debug('Question to process@517: 〚%s〛', question_text)
                normalized_phrase_1 = self.normalize_person(question_text)

                # ВЕТКА P(1)Q

                premises = []
                rels = []

                #matches_2 = self.relevancy_detector.match2(normalized_phrase_1, memory_phrases, score_threshold=self.pqa_rel_threshold)
                #for premise, premise_rel in matches_2:
                premises0, rels0 = self.relevancy_detector.get_most_relevant(normalized_phrase_1, memory_phrases, nb_results=2)
                for premise, premise_rel in zip(premises0, rels0):
                    if premise_rel >= self.pqa_rel_threshold:
                        # В базе знаний нашелся релевантный факт.
                        premises.append(premise)
                        rels.append(premise_rel)
                        self.logger.debug('KB lookup@533: query=〚%s〛 premise=〚%s〛 rel=%5.3f', normalized_phrase_1, premise, premise_rel)

                phrase_modality, phrase_person, raw_tokens = self.modality_model.get_modality(question_text, self.text_utils)

                if len(premises) == 0 and phrase_modality == ModalityDetector.question:
                    # Сценарии P(0)Q и P(1)Q не смогли сгенерировать ответ.
                    # Пробуем сценарий P(2)Q
                    q = normalized_phrase_1
                    matches = self.p2q_scorer.match2(q, memory_phrases)
                    #for premise1, premise2, total_score in matches[:10]:
                    #    print('DEBUG@499 [{:6.3f}]  premise1=〚{}〛   premise2=〚{}〛'.format(total_score, premise1, premise2))
                    # Будем фильтровать найденные тройки через closure-модель
                    p2q_texts = []
                    for premise1, premise2, total_score in matches[:100]:
                        parts = []
                        for p in [premise1, premise2, q]:
                            p = p[0].upper() + p[1:]
                            if p[-1] not in '.!;?…':
                                parts.append(p+'.')
                            else:
                                parts.append(p)
                        merged_context = ' '.join(parts)
                        p2q_texts.append(merged_context)
                    p2q_scores = self.closure_detector.score_contexts(p2q_texts)
                    p2q_filtered_matches = []
                    for (premise1, premise2, score1), score2 in zip(matches, p2q_scores):
                        score = score1 * score2
                        if score > self.pqa_rel_threshold:
                            p2q_filtered_matches.append((premise1, premise2, q, score))

                    p2q_filtered_matches = sorted(p2q_filtered_matches, key=lambda z: -z[3])
                    for premise1, premise2, q, score in p2q_filtered_matches[:2]:
                        proba = score * p_interp
                        p2q_responses = self.generate_p2qa_reply(dialog, premise1, premise2, q, proba, interpretation)
                        responses.extend(p2q_responses)


                dodged = False
                if len(premises) > 0:
                    # Нашлись релевантные предпосылки, значит мы попадем в ветку PQA.
                    # С заданной вероятностью переходим на отдельную ветку "уклонения от ответа":
                    if random.random() < profile.p_dodge1:
                        for interpretation, p_interp in all_interpretations:
                            dodge_replies = self.generate_dodge_reply(dialog, interpretation, p_interp)
                            if dodge_replies:
                                responses.extend(dodge_replies)
                                dodged = True
                                premises.clear()
                                rels.clear()

                # С помощью каждого найденного факта (предпосылки) будем генерировать варианты ответа, используя режим PQA читчата
                for premise, premise_relevancy in zip(premises, rels):
                    if premise_relevancy >= self.pqa_rel_threshold:  # Если найденная в БД предпосылка достаточно релевантна вопросу...
                        confab_premises.append(([premise], premise_relevancy*question_w, 'knowledgebase'))

                if len(confab_premises) == 0 and use_confabulation and not dodged:
                    if phrase_person != '2':  # не будем выдумывать факты про собеседника!
                        # В базе знаний ничего релевантного не нашлось.
                        # Мы можем а) сгенерировать ответ с семантикой "нет информации" б) заболтать вопрос в) придумать факт и уйти в ветку PQA
                        # Используем заданные константы профиля для выбора ветки.
                        x = random.random()
                        if x < profile.p_confab:
                            # Просим конфабулятор придумать варианты предпосылок.
                            #confabul_context = [interpretation]  #[self.flip_person(interpretation)]
                            # TODO - первый запуск делать с num_return_sequences=10, второй с num_return_sequences=100
                            confabulations = self.confabulator.generate_confabulations(question=interpretation, num_return_sequences=10)
                            self.logger.debug('Confabulation@604: context=〚%s〛 outputs=〚%s〛', interpretation, format_outputs(confabulations))

                            for confab_text in confabulations:
                                score = 1.0

                                # Может быть несколько предпосылок, поэтому бьем на клаузы.
                                premises = self.text_utils.split_clauses(confab_text)

                                # Понижаем достоверность конфабуляций, относящихся к собеседнику.
                                for premise in premises:
                                    words = self.text_utils.tokenize(premise)
                                    if any((w.lower() == 'ты') for w in words):
                                        score *= 0.5

                                confab_premises.append((premises, score, 'confabulation'))

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
                                fx, rels = self.synonymy_detector.get_most_similar(confab_premise, memory_phrases, nb_results=1)
                                if fx:
                                    memory_phrase = fx[0]
                                    rel = rels[0]
                                else:
                                    memory_phrase = None
                                    rel = 0.0

                                if rel > 0.5:
                                    if memory_phrase != confab_premise:
                                        self.logger.debug('Synonymy@647 text1=〚%s〛 text2=〚%s〛 score=%5.3f', confab_premise, memory_phrase, rel)

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

                            # 14.09.2022 меняем грамматическое лицо, чтобы факт шел от лица собеседника
                            premise_facts2 = [self.interpreter.flip_person(fact, self.text_utils) for fact in premise_facts]

                            pqa_responses = self.generate_pqa_reply(dialog,
                                                                    interpretation,
                                                                    p_interp,
                                                                    processed_chitchat_contexts,
                                                                    premise_facts=premise_facts2,
                                                                    premises_proba=total_proba,
                                                                    unmapped_confab_facts=unmapped_confab_facts)
                            responses.extend(pqa_responses)

                if len(responses) == 0 and phrase_modality == ModalityDetector.question:
                    # Собеседник задал вопрос, но мы не смогли ответить на него с помощью имеющейся в базе знаний
                    # информации, и ветка конфабуляции тоже не смогла ничего выдать. Остается 2 пути: а) ответить "нет информации" б) заболтать вопрос
                    dodged = False
                    if profile.p_dodge2:
                        # Пробуем заболтать вопрос.
                        dodge_responses = self.generate_dodge_reply(dialog, interpretation, p_interp)
                        if dodge_responses:
                            responses.extend(dodge_responses)
                            dodged = True
                    if not dodged:
                        # Пробуем сгенерировать ответ с семантикой "нет информации"
                        noinfo_responses = self.generate_noinfo_reply(dialog, interpretation, p_interp)
                        if noinfo_responses:
                            responses.extend(noinfo_responses)

            if len(questionx) == 0:
                # Генеративный читчат делает свою основную работу - генерирует ответную реплику.
                chitchat_responses = self.generate_chitchat_reply(dialog, interpretation, p_interp)
                responses.extend(chitchat_responses)

        # Генерируем отложенные реплики.
        responses2 = [response for response in responses if response.generation_promise.is_generated()]
        chitchat_promises = [r for r in responses if r.generation_promise.is_promised()]
        responses3 = self.generate_promised_responses(chitchat_promises)

        # Генерация вариантов ответной реплики закончена.
        responses = responses2 + responses3

        # Делаем оценку сгенерированных реплик - насколько хорошо они вписываются в текущий контекст диалога
        chitchat_context0 = dialog.construct_chitchat_context(last_utterance_interpretation=None, last_utterance_labels=None, include_commands=False)
        #px_entail = self.entailment.predictN(' | '.join(chitchat_context0), [r.get_text() for r in responses])
        px_entail = self.chitchat.score_dialogues([(chitchat_context0 + [r.get_text()]) for r in responses])

        for r, p_entail in zip(responses, px_entail):
            r.set_p_entail(p_entail)

        # Сортируем по убыванию скора
        responses = sorted(responses, key=lambda z: -z.get_proba())

        self.logger.debug('%d responses generated for input_message=〚%s〛 interlocutor="%s" bot="%s":', len(responses), dialog.get_last_message().get_text(), interlocutor, profile.get_id())
        table = [['i', 'text', 'p_entail', 'score', 'algo', 'context', 'confabulations']]
        for i, r in enumerate(responses, start=1):
            table.append((str(i), r.get_text(), '{:5.3f}'.format(r.p_entail), '{:5.3f}'.format(r.get_proba()), r.get_algo(), r.get_context(), format_confabulations_list(r.get_confabulated_facts())))

        for x in terminaltables.AsciiTable(table).table.split('\n'):
            logging.debug('%s', x)

        if len(responses) == 0:
            self.logger.error('No response generated in context: message=〚%s〛 interlocutor="%s" bot="%s"',
                             dialog.get_last_message().get_text(), interlocutor, profile.get_id())
            self.print_dialog(dialog)
            return []

        # Выбираем лучший response, запоминаем интерпретацию последней фразы в истории диалога.
        # 16.02.2022 Идем по списку сгенерированных реплик, проверяем реплику на отсутствие противоречий или заеданий.
        # Если реплика плохая - отбрасываем и берем следующую в сортированном списке.
        self_interpretation = None
        for best_response in responses:
            is_good_reply = True

            if best_response.prev_utterance_interpretation is not None:
                # Если входная обрабатываемая реплика содержит какой-то факт, то его надо учитывать сейчас при поиске
                # релевантных предпосылок. Но так как мы еще не уверены, что именно данный вариант интерпретации входной
                # реплики правильный, то просто соберем временный список с добавленной интерпретацией.
                input_assertions, input_questions = split_message_text(best_response.prev_utterance_interpretation, self.text_utils)
                memory_phrases2 = list(memory_phrases)
                for assertion_text in input_assertions:
                    fact_text2 = self.flip_person(assertion_text)
                    memory_phrases2.append((fact_text2, '', '(((tmp@741)))'))

                # Вполне может оказаться, что наша ответная реплика - краткая, и мы должны попытаться восстановить
                # полную реплику перед семантическими и прагматическими проверками.
                prevm = best_response.prev_utterance_interpretation # dialog.get_last_message().get_interpretation()
                if prevm is None:
                    prevm = dialog.get_last_message().get_text()
                interpreter_context = prevm + ' | ' + best_response.get_text()
                self_interpretation = self.interpreter.interpret([z.strip() for z in interpreter_context.split('|')], num_return_sequences=1)[0]
                self.logger.debug('Self interpretation@750: context=〚%s〛 output=〚%s〛', interpreter_context, self_interpretation)

                self_assertions, self_questions = split_message_text(self_interpretation, self.text_utils)
                for question_text in self_questions:
                    # Реплика содержит вопрос. Проверим, что мы ранее не задавали такой вопрос, и что
                    # мы не знаем ответ на этот вопрос. Благодаря этому бот не будет спрашивать снова то, что уже
                    # спрашивал или что он просто знает.
                    self.logger.debug('Question to process@757: 〚%s〛', question_text)
                    premises, rels = self.relevancy_detector.get_most_relevant(question_text, memory_phrases2, nb_results=1)
                    if len(premises) > 0:
                        premise = premises[0]
                        rel = rels[0]
                        if rel >= self.pqa_rel_threshold:
                            self.logger.debug('KB lookup@763: query=〚%s〛 premise=〚%s〛 rel=%f', question_text, premise, rel)
                            # Так как в БД найден релевантный факт, то бот уже знает ответ на этот вопрос, и нет смысла задавать его
                            # собеседнику снова.
                            is_good_reply = False
                            self.logger.debug('Output response 〚%s〛 contains a question 〚%s〛 with known answer, so skipping it @759', best_response.get_text(), question_text)
                            break

                if not is_good_reply:
                    continue

                # проверяем по БД, нет ли противоречий с утвердительной частью.
                # Генерации реплики, сделанные из предпосылки в БД, не будем проверять.
                if best_response.get_algo() != 'pqa_response':
                    for assertion_text in self_assertions:
                        # Ищем релевантный факт в БД
                        premises, rels = self.relevancy_detector.get_most_relevant(assertion_text, memory_phrases2, nb_results=1)
                        if len(premises) > 0:
                            premise = premises[0]
                            rel = rels[0]
                            if rel >= self.pqa_rel_threshold:
                                self.logger.debug('KB lookup@792: query=〚%s〛 premise=〚%s〛 rel=%f', assertion_text, premise, rel)

                                # Формируем запрос на генерацию ответа через gpt читчата...
                                chitchat_context = premise[0].upper() + premise[1:] #+ '. ' + assertion_text + '?'
                                if premise[-1] not in '!;.':
                                    chitchat_context += '.'
                                chitchat_context += ' ' + assertion_text + '?'

                                chitchat_outputs = self.chitchat.generate_chitchat(context_replies=[chitchat_context], num_return_sequences=5)
                                self.logger.debug('PQA@797: context=〚%s〛 outputs=〚%s〛', chitchat_context, format_outputs(chitchat_outputs))
                                for chitchat_output in chitchat_outputs:
                                    # Заглушка - ищем отрицательные частицы
                                    words = self.text_utils.tokenize(chitchat_output)
                                    if any((w.lower() in ['нет', 'не']) for w in words):
                                        is_good_reply = False
                                        self.logger.debug('Output response 〚%s〛 contains assertion 〚%s〛 which contradicts the knowledge base', best_response.get_text(), assertion_text)
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
            self.store_new_fact(fact_text2, '(((dialog@820)))', dialog, profile, facts)

        # Если при генерации ответной реплики использован вымышленный факт, то его надо запомнить в базе знаний.
        if best_response.get_confabulated_facts():
            for f in best_response.get_confabulated_facts():
                self.store_new_fact(f, '(((confabulation@825)))', dialog, profile, facts)

        # Добавляем в историю диалога выбранную ответную реплику
        self.logger.debug('Response for input message 〚%s〛 from interlocutor="%s": text=〚%s〛 self_interpretation=〚%s〛 algorithm="%s" score=%5.3f', dialog.get_last_message().get_text(),
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

    def generate_promised_responses(self, chitchat_promises):
        responses = []

        # Делаем прогон всех контекстов генерации одним батчем:
        num_return_sequences = 2
        chitchat_outputs_batch = self.chitchat.generate_chitchat_batch([promise.generation_promise.chitchat_generation_context for promise in chitchat_promises], num_return_sequences=num_return_sequences)

        for ipromise, promise in enumerate(chitchat_promises):
            chitchat_outputs = chitchat_outputs_batch[ipromise*num_return_sequences: (ipromise+1)*num_return_sequences]
            for chitchat_output in chitchat_outputs:
                # Оценка синтаксической валидности реплики
                p_valid = 1.0  # self.syntax_validator.is_valid(chitchat_output, text_utils=self.text_utils)
                if p_valid < self.min_nonsense_threshold:
                    self.logger.debug('Nonsense detector@905: text=〚%s〛 p=%5.3f', chitchat_output, p_valid)
                    continue

                explicit_response = GeneratedResponse(promise.algo,
                                           prev_utterance_interpretation=promise.prev_utterance_interpretation,
                                           generation_promise=ResponseGenerationPromise.make_explicit(chitchat_output),
                                           p=p_valid * promise.p,
                                           confabulated_facts=promise.confabulated_facts,
                                           context=' | '.join(promise.generation_promise.chitchat_generation_context))
                responses.append(explicit_response)

        return responses

    def generate_dodge_reply(self, dialog, interpretation, p_interp):
        message_labels = ['уклониться от ответа']
        chitchat_context = dialog.construct_chitchat_context(interpretation, message_labels)
        chitchat_promise = ResponseGenerationPromise.make_promise(chitchat_context)

        responses = []
        responses.append(GeneratedResponse('dodge_response@924',
                                           prev_utterance_interpretation=interpretation,
                                           generation_promise=chitchat_promise,
                                           p=p_interp,
                                           confabulated_facts=None,
                                           context=' | '.join(chitchat_context)))
        return responses

    def generate_noinfo_reply(self, dialog, interpretation, p_interp):
        message_labels = ['нет информации']
        chitchat_context = dialog.construct_chitchat_context(interpretation, message_labels)
        chitchat_promise = ResponseGenerationPromise.make_promise(chitchat_context)

        responses = []
        responses.append(GeneratedResponse('noinfo_response@938',
                                           prev_utterance_interpretation=interpretation,
                                           generation_promise=chitchat_promise,
                                           p=p_interp,
                                           confabulated_facts=None,
                                           context=' | '.join(chitchat_context)))
        return responses

    def generate_chitchat_reply(self, dialog, interpretation, p_interp):
        message_labels = []
        chitchat_context = dialog.construct_chitchat_context(interpretation, message_labels)
        chitchat_promise = ResponseGenerationPromise.make_promise(chitchat_context)

        responses = []
        responses.append(GeneratedResponse('chitchat_response@952',
                                           prev_utterance_interpretation=interpretation,
                                           generation_promise=chitchat_promise,
                                           p=p_interp,
                                           confabulated_facts=None,
                                           context=' | '.join(chitchat_context)))
        return responses

    def generate_p0qa_reply(self, dialog, prev_utterance_interpretation, reply_text, rel_p0q):
        """
        Генерация ответа в ситуации, когда текст входной реплики содержит всю необходимую для ответа информацию.

        Например:

        - 2+2=?
        - 4
        """
        chitchat_context = [reply_text]
        chitchat_promise = ResponseGenerationPromise.make_promise(chitchat_context)

        responses = []
        responses.append(GeneratedResponse('p0qa_response@973',
                                           generation_promise=chitchat_promise,
                                           prev_utterance_interpretation=prev_utterance_interpretation,
                                           p=1.0,
                                           confabulated_facts=None,
                                           context=' | '.join(chitchat_context)))
        return responses

    def generate_p2qa_reply(self, dialog, premise1, premise2, question, proba, interpretation):
        """
        Генерация ответа на основе двух предпосылок и вопроса, т.е. P(2)QA.

        Например:

        premise1=Сократ - греческий философ.
        premise2=Все философы смертны.
        question=Смертен ли Сократ?
        """
        context = []
        for p in [premise1, premise2]:
            p = self.flip_person(p)
            if p[-1] not in '.!?;':
                p += '.'
            context.append(p)
        context.append(question)
        context_str = ' '.join(context)

        chitchat_context = [context_str]
        chitchat_promise = ResponseGenerationPromise.make_promise(chitchat_context)

        responses = []
        responses.append(GeneratedResponse('p2qa_response@1004',
                                           prev_utterance_interpretation=interpretation,
                                           generation_promise=chitchat_promise,
                                           p=proba,
                                           confabulated_facts=None,
                                           context=' | '.join(chitchat_context)))
        return responses

    def generate_pqa_reply(self, dialog, interpretation, p_interp, processed_chitchat_contexts, premise_facts, premises_proba, unmapped_confab_facts):
        # Пробуем сгенерировать ответ, опираясь на найденные в базе знаний предпосылки и заданный собеседником вопрос.
        # 07.03.2022 ограничиваем длину контекста
        chitchat_context = dialog.construct_chitchat_context(interpretation, premise_facts, max_depth=1)
        chitchat_context_str = '|'.join(chitchat_context)
        responses = []

        if chitchat_context_str not in processed_chitchat_contexts:
            processed_chitchat_contexts.add(chitchat_context_str)
            chitchat_promise = ResponseGenerationPromise.make_promise(chitchat_context)
            responses.append(GeneratedResponse('pqa_response@1022',
                                               prev_utterance_interpretation=interpretation,
                                               generation_promise=chitchat_promise,
                                               p=p_interp * premises_proba,
                                               confabulated_facts=None,
                                               context=' | '.join(chitchat_context)))

        return responses


def split_message_text(message, text_utils):
    assertions = []
    questions = []

    for clause in text_utils.split_clauses(message):
        if clause.endswith('?'):
            questions.append(clause)
        else:
            assertions.append(clause)

    return assertions, questions


def format_confabulations_list(confabulations):
    sx = []
    if confabulations:
        for s in confabulations:
            if s[-1] not in '.!?':
                sx.append(s+'.')
            else:
                sx.append(s)

    return ' '.join(sx)


def format_outputs(outputs):
    if len(outputs) == 1:
        return outputs[0]

    sx = []
    for i, s in enumerate(outputs, start=1):
        sx.append('[{}] {}'.format(i, s))
    return ' | '.join(sx)

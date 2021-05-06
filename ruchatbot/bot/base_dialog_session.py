# -*- coding: utf-8 -*-

import datetime
import collections

from collections import deque
from ruchatbot.bot.running_scenario import RunningDialogStatus


class BaseDialogSession(object):
    """
    Хранилище оперативных данных для диалоговой сессии с одним собеседником.
    Персистентность реализуется производными классами.
    """
    def __init__(self, bot_id, interlocutor, facts_storage):
        """
        Инициализация новой диалоговой сессии для нового собеседника.
        :param bot_id: уникальный строковый идентификатор бота
        :param interlocutor: строковый идентификатор собеседника.
        :param facts_storage: объект класса BaseFactsStorage, реализующего
         чтение и сохранение фактов.
        """
        self.bot_id = bot_id
        self.interlocutor = interlocutor
        self.start_time = datetime.datetime.now() # datetime.datetime.now().timestamp()
        self.last_activity_time = self.start_time
        self.facts_storage = facts_storage
        self.conversation_history = []  # все фразы беседы, экземпляры InterpretedPhrase
        self.output_b_index = -1  # индекс последней отданной наружу B-фразы в списке conversation_history

        self.activated_rules = []  # правила-обработчики, сработавшие (рекурсивно) в ходе обработки реплики собеседника

        self.status = None  # экземпляр производного от RunningDialogStatus класса,
                            # если выполняется вербальная форма или сценарий
        self.deferred_running_items = deque()
        self.slots = dict()  # переменные состояния
        self.started_scenarios = set()  # какие сценарии запускались в этой сессии
        self.premise_not_found_counter = 0  # для отладки: сколько раз не удалось ответить на вопрос с помощью фактов в БЗ
        self.order_not_handled_counter = 0  # для отладки: сколько раз не удалось обработать императив
        self.cannot_answer_counter = 0  # сколько раз не удалось выдать ответ, в том числе с помощью правил в no-info модели
        self.nb_commented_contradictions = 0  # сколько раз попали в ветку "а я ...."
        self.actor_say_hits = collections.Counter()  # счетчики срабатывания акторов say в привязке к id

    def get_interlocutor(self):
        return self.interlocutor

    def get_bot_id(self):
        return self.bot_id

    def before_processing_new_input(self):
        """
        Вызывается всякий раз перед тем, как начинает обрабатываться новая входная реплика
        собеседника.
        """
        self.last_activity_time = datetime.datetime.now()

    def get_start_time(self):
        return self.start_time

    def get_last_activity_time(self):
        return self.last_activity_time

    def add_phrase_to_history(self, interpreted_phrase):
        self.conversation_history.append(interpreted_phrase)

    def add_output_phrase(self, interpreted_phrase):
        phrase_text = interpreted_phrase.raw_phrase

        # Сколько B-реплик, ожидающих выдачи, уже накопилось в истории.
        nb_last_b = 0
        for history_item in self.conversation_history[::-1]:
            if not history_item.is_bot_phrase:
                break
            else:
                nb_last_b += 1

        if nb_last_b == 0:
            # Так как ожидающих выдачи B-фраз нет, то просто добавляем новую запись в историю.
            self.conversation_history.append(interpreted_phrase)
        else:
            # 23-12-2020 сортируем фразы так, чтобы вопросы были в конце.
            if phrase_text.endswith('?'):
                self.conversation_history.append(interpreted_phrase)
            else:
                if self.conversation_history[-1].raw_phrase.endswith('?'):
                    # вставляем добавляемый не-вопрос ПЕРЕД хвостовым вопросом.
                    self.conversation_history.insert(len(self.conversation_history) - 1, interpreted_phrase)
                else:
                    self.conversation_history.append(interpreted_phrase)

    def insert_output_phrase(self, interpreted_phrase):
        phrase_text = interpreted_phrase.raw_phrase

        # Сколько B-реплик, ожидающих выдачи, уже накопилось в истории.
        nb_last_b = 0
        for history_item in self.conversation_history[::-1]:
            if not history_item.is_bot_phrase:
                break
            else:
                nb_last_b += 1

        if nb_last_b > 0:
            if phrase_text.endswith('?'):
                # Вопросы должны быть в хвосте
                self.conversation_history.append(interpreted_phrase)
            else:
                self.conversation_history.insert(len(self.conversation_history)-nb_last_b, interpreted_phrase)
        else:
            self.conversation_history.append(interpreted_phrase)

    def get_output_buffer_phrase(self):
        if self.conversation_history:
            last_h = self.conversation_history[-1]
            if last_h.is_bot_phrase:
                return last_h.raw_phrase
        return None

    def extract_from_buffer(self):
        """
        Извлекает и возвращает самую старую готовую фразу
        из буфера ответов.
        :return: ответ бота или пустая строка, если буфер ответов пуст.
        """

        # сбрасываем инфу о сработавших правилах, так как явно закончилась обработка предыдущей
        # реплики собеседника.
        self.activated_rules = []

        # Ищем первую фразу после индекса output_b_index
        for i in range(self.output_b_index+1, len(self.conversation_history)):
            if self.conversation_history[i].is_bot_phrase:
                self.output_b_index = i
                return self.conversation_history[i].raw_phrase

        # В истории нет невыданных B-фраз
        return ''

    def rule_activated(self, rule):
        self.activated_rules.append(rule)

    def is_rule_activated(self, rule):
        return rule in self.activated_rules

    def get_interlocutor_phrases(self, questions=True, assertions=True, last_nb=10):
        """
        Возвращается список последних last_nb реплик, которые произносил
        собеседник. Возвращается список из кортежей (phrase, timegap), где timegap - сколько шагов
        тому назад была произнесена фраза.
        """
        reslist = []
        for timegap, item in enumerate(self.conversation_history[::-1]):
            if not item.is_bot_phrase:
                if questions and item.is_question:
                    # добавляем вопрос
                    reslist.append((item, timegap))
                elif assertions and not item.is_question:
                    # добавляем не-вопрос
                    reslist.append((item, timegap))
        return reslist[:last_nb]

    def extract_last_chitchat_context_pair(self):
        """Возвращает самую последнюю реплику собеседника и предыдущую реплику бота"""
        i = len(self.conversation_history)-1
        while i >= 0:
            if not self.conversation_history[i].is_bot_phrase:
                h_phrase = self.conversation_history[i]
                b_phrase = None
                if i > 0:
                    if self.conversation_history[i-1].is_bot_phrase:
                        b_phrase = self.conversation_history[i-1]

                return b_phrase, h_phrase

            i -= 1

        return None, None

    def count_bot_phrase(self, phrase_str):
        """Вернет, сколько раз бот уже произносил фразу phrase_str"""
        n = 0
        for item in self.conversation_history:
            if item.is_bot_phrase and item.interpretation == phrase_str:
                n += 1

        return n

    def get_bot_phrases(self):
        res = []
        for item in self.conversation_history:
            if item.is_bot_phrase:
                res.append(item.interpretation)
        return res

    def count_interlocutor_phrases(self):
        n = 0
        for item in self.conversation_history:
            if not item.is_bot_phrase:
                n += 1
        return n

    def get_all_phrases(self):
        res = []
        for item in self.conversation_history:
            res.append(item.interpretation)
        return res

    def get_last_interlocutor_utterance(self):
        for item in self.conversation_history[::-1]:
            if not item.is_bot_phrase:
                return item
        return None

    def get_last_bot_utterance(self):
        for item in self.conversation_history[::-1]:
            if item.is_bot_phrase:
                return item
        return None

    def get_last_utterance(self):
        return self.conversation_history[-1] if len(self.conversation_history) > 0 else None

    def select_answer_buffer_bs(self):
        output_bs = []
        for history_item in self.conversation_history[::-1]:
            if not history_item.is_bot_phrase:
                break
            else:
                assert(len(history_item.raw_phrase) > 0)
                output_bs.insert(0, history_item.raw_phrase)

        return output_bs

    def select_prev_consequent_bs(self):
        """Вернем список с текстами последних подряд идущих реплик бота"""
        bs = []
        for item in self.conversation_history[::-1]:
            if item.is_bot_phrase:
                bs.append(item.raw_phrase)
            else:
                break
        return bs

    def count_prev_consequent_b(self):
        """Количество подряд идущих справа (т.е. в конце сессии) B-фраз"""
        nb = 0
        for item in self.conversation_history[::-1]:
            if item.is_bot_phrase:
                nb += 1
            else:
                break

        return nb

    def set_status(self, new_status):
        if new_status is None:
            # Если в стеке отложенных сценариев есть что-то, запускаем его.
            if len(self.deferred_running_items) > 0:
                self.status = self.deferred_running_items.pop()
            else:
                self.status = None
        else:
            assert isinstance(new_status, RunningDialogStatus)
            self.status = new_status
            self.started_scenarios.add(new_status.get_name())

    def call_scenario(self, running_scenario):
        assert isinstance(running_scenario, RunningDialogStatus)
        self.started_scenarios.add(running_scenario.get_name())
        if self.status is None:
            self.status = running_scenario
        else:
            self.deferred_running_items.append(self.status)
            self.status = running_scenario

    def exit_scenario(self):
        if len(self.deferred_running_items) > 0:
            self.status = self.deferred_running_items.pop()
        else:
            self.status = None

    def defer_status(self, new_status):
        if not isinstance(new_status, RunningDialogStatus):
            raise AssertionError()

        self.deferred_running_items.append(new_status)

    def raise_deferred_scenario(self, scenario_name):
        i = [x.get_name() for x in self.deferred_running_items].index(scenario_name)
        running_scenario = self.deferred_running_items[i]
        del self.deferred_running_items[i]
        self.deferred_running_items.insert(0, self.status)
        self.status = running_scenario

    def form_executed(self):
        self.status = None

    def get_status(self):
        return self.status

    def is_deferred_scenario(self, scenario_name):
        return scenario_name in (x.get_name() for x in self.deferred_running_items)

    def get_scenario_stack_depth(self):
        """Вернет количество сценариев в сессии - один текущий и еще сколько-то в стеке отложенных"""
        n = 0
        if self.status:
            n += 1
            n += len(self.deferred_running_items)
        return n

    def list_scenario_stack(self):
        names = []
        if self.status:
            names.append('0:{}@{}'.format(self.status.get_name(), self.status.get_current_step_name()))
            for depth, item in enumerate(self.deferred_running_items, start=1):
                names.append('-{}:{}@{}'.format(depth, item.get_name(), item.get_current_step_name()))
        return ' '.join(names)

    def cancel_all_running_items(self):
        self.deferred_running_items = []
        self.status = None

    def get_slot(self, slot_name):
        return self.slots.get(slot_name, '')

    def set_slot(self, slot_name, slot_value):
        self.slots[slot_name] = slot_value

    def reset_history(self):
        self.conversation_history = []
        self.slots.clear()
        self.activated_rules.clear()
        self.deferred_running_items.clear()
        self.output_b_index = -1
        self.started_scenarios.clear()
        self.premise_not_found_counter = 0
        self.order_not_handled_counter = 0
        self.cannot_answer_counter = 0
        self.nb_commented_contradictions = 0

    def set_causal_clause(self, interpreted_phrase):
        for item in self.conversation_history[::-1]:
            if item.causal_interpretation_clause is not None or not item.is_bot_phrase:
                break
            else:
                item.causal_interpretation_clause = interpreted_phrase

    def purge_bot_phrases(self):
        # Собираем все последние фразы бота
        purged_phrases = []
        for item in self.conversation_history[::-1]:
            if not item.is_bot_phrase:
                break
            else:
                purged_phrases.append(item)

        if purged_phrases:
            # Удаляем все собранные фраз бота из истории.
            self.conversation_history = self.conversation_history[:-len(purged_phrases)]

        return purged_phrases

    def premise_not_found(self):
        self.premise_not_found_counter += 1

    def order_not_handled(self):
        self.order_not_handled_counter += 1

    def get_session_stat(self):
        lines = []
        lines.append('Scenarios: {}'.format(', '.join(self.started_scenarios)))
        lines.append('premise_not_found_counter={}'.format(self.premise_not_found_counter))
        lines.append('order_not_handled_counter={}'.format(self.order_not_handled_counter))
        return lines

    def scenario_already_run(self, scenario_name):
        return scenario_name in self.started_scenarios

    def is_empty(self):
        return len(self.conversation_history) == 0

    def get_actor_say_hits(self, actor_say_id):
        return self.actor_say_hits[actor_say_id]

    def actor_say_hit(self, actor_say_id):
        self.actor_say_hits[actor_say_id] += 1

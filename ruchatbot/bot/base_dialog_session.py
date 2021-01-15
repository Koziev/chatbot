# -*- coding: utf-8 -*-

from collections import deque


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
        self.facts_storage = facts_storage
        self.conversation_history = []  # все фразы беседы, экземпляры InterpretedPhrase
        self.output_b_index = -1  # индекс последней отданной наружу B-фразы в списке conversation_history

        self.activated_rules = []  # правила-обработчики, сработавшие (рекурсивно) в ходе обработки реплики собеседника

        self.status = None  # экземпляр производного от RunningDialogStatus класса,
                            # если выполняется вербальная форма или сценарий
        self.deferred_running_items = deque()

        self.slots = dict()  # переменные состояния

    def get_interlocutor(self):
        return self.interlocutor

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
        """Количество подряд идущих справа B-фраз"""
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
            self.status = new_status

    def call_scenario(self, running_scenario):
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
        self.deferred_running_items.append(new_status)

    def form_executed(self):
        self.status = None

    def get_status(self):
        return self.status

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
            names.append('0:{}'.format(self.status.get_name()))
            for depth, item in enumerate(self.deferred_running_items, start=1):
                names.append('-{}:{}'.format(depth, item.get_name()))
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
        #self.answer_buffer.clear()
        self.slots.clear()
        self.activated_rules.clear()
        self.deferred_running_items.clear()
        self.output_b_index = -1

    def set_causal_clause(self, interpreted_phrase):
        for item in self.conversation_history[::-1]:
            if item.causal_interpretation_clause is not None or not item.is_bot_phrase:
                break
            else:
                item.causal_interpretation_clause = interpreted_phrase


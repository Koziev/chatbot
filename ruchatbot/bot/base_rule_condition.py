# -*- coding: utf-8 -*-

import logging
import re

from ruchatbot.bot.keyword_matcher import KeywordMatcher


class BaseRuleCondition(object):
    def __init__(self):
        self.condition = None

    @staticmethod
    def from_yaml(condition_yaml):
        # Пока заглушка, потом надо сделать компиляцию для ускорения проверки в check_condition
        res = BaseRuleCondition()
        res.condition = condition_yaml
        return res

    def get_key(self):
        return u';'.join(k+'|'+v for (k, v) in self.condition.items())

    def check_text(self, input_text, etalon_texts, bot, session, interlocutor, interpreted_phrase, answering_engine):
        text_utils = answering_engine.get_text_utils()
        word_embeddings = answering_engine.get_word_embeddings()

        input_text = text_utils.wordize_text(input_text)
        etalons = list((text_utils.wordize_text(etalon), None, None) for etalon in etalon_texts)

        syn = answering_engine.get_synonymy_detector()
        best_etalon, best_sim = syn.get_most_similar(input_text,
                                                     etalons,
                                                     text_utils,
                                                     word_embeddings,
                                                     nb_results=1)
        return best_sim >= syn.get_threshold()

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        if u'intent' in self.condition:
            return self.condition[u'intent'] == interpreted_phrase.intent

        elif u'state' in self.condition:
            slot_name = self.condition['state']['slot']
            slot_mask = self.condition['state']['regex_mask']
            slot_value = session.get_slot(slot_name)
            return slot_value and re.match(slot_mask, slot_value)

        elif u'text' in self.condition:
            if isinstance(self.condition[u'text'], list) or isinstance(self.condition[u'text'], str):
                # TODO: Эти приготовления будут выполняться при каждом запуске правила, надо бы
                # вынести их в конструктор правила.
                if isinstance(self.condition[u'text'], list):
                    etalons = self.condition[u'text']
                else:
                    etalons = [self.condition[u'text']]

                input_text = interpreted_phrase.interpretation
                return self.check_text(input_text, etalons, bot, session, interlocutor, interpreted_phrase, answering_engine)
            else:
                logging.error(u'Conditional statement "%s" can not be processed', self.condition[u'text'])
                raise NotImplementedError()

        elif u'raw_text' in self.condition:
            if isinstance(self.condition[u'raw_text'], list) or isinstance(self.condition[u'raw_text'], str):
                if isinstance(self.condition[u'raw_text'], list):
                    etalons = self.condition[u'raw_text']
                else:
                    etalons = [self.condition[u'raw_text']]

                input_text = interpreted_phrase.raw_phrase
                return self.check_text(input_text, etalons, bot, session, interlocutor, interpreted_phrase, answering_engine)
            else:
                logging.error(u'Conditional statement "%s" can not be processed', self.condition[u'text'])
                raise NotImplementedError()

        elif u'prev_bot_text' in self.condition:
            # Проверяем, что последняя фраза бота была похожа на указанный эталон
            if isinstance(self.condition[u'prev_bot_text'], list) or isinstance(self.condition[u'prev_bot_text'], str):
                if isinstance(self.condition[u'prev_bot_text'], list):
                    etalons = self.condition[u'prev_bot_text']
                else:
                    etalons = [self.condition[u'prev_bot_text']]

                input_text = session.get_last_bot_utterance().interpretation
                return self.check_text(input_text, etalons, bot, session, interlocutor, interpreted_phrase, answering_engine)
            else:
                logging.error(u'Conditional statement "%s" can not be processed', self.condition[u'text'])
                raise NotImplementedError()

        elif u'keyword' in self.condition:
            if isinstance(self.condition[u'keyword'], list) or isinstance(self.condition[u'keyword'], str):
                if isinstance(self.condition[u'keyword'], list):
                    etalons = self.condition[u'keyword']
                else:
                    etalons = [self.condition[u'keyword']]

                matchers = [KeywordMatcher.from_string(etalon) for etalon in etalons]

                for matcher in matchers:
                    if not matcher.match(interpreted_phrase):
                        return False

                return True
        elif u'regex' in self.condition:
            rx_str = self.condition['regex']
            return re.search(rx_str, interpreted_phrase.interpretation, flags=re.IGNORECASE)
        else:
            raise NotImplementedError()

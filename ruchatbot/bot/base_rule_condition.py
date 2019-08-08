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

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        if u'intent' in self.condition:
            return self.condition[u'intent'] == interpreted_phrase.intent
        elif u'text' in self.condition:
            if isinstance(self.condition[u'text'], list) or isinstance(self.condition[u'text'], str):
                text_utils = answering_engine.get_text_utils()
                word_embeddings = answering_engine.get_word_embeddings()

                # TODO: Эти приготовления будут выполняться при каждом запуске правила, надо бы
                # вынести их в конструктор правила.
                if isinstance(self.condition[u'text'], list):
                    etalons = self.condition[u'text']
                else:
                    etalons = [self.condition[u'text']]

                etalons = list((text_utils.wordize_text(etalon), None, None) for etalon in etalons)

                input_text = text_utils.wordize_text(interpreted_phrase.interpretation)
                syn = answering_engine.get_synonymy_detector()
                best_etalon, best_sim = syn.get_most_similar(input_text,
                                                             etalons,
                                                             text_utils,
                                                             word_embeddings,
                                                             nb_results=1)

                return best_sim >= syn.get_threshold()
            else:
                logging.error(u'Conditonal statement "%s" can not be processed', self.condition[u'text'])
                raise NotImplementedError()
        elif u'keyword' in self.condition:
            if isinstance(self.condition[u'keyword'], list) or isinstance(self.condition[u'keyword'], str):
                text_utils = answering_engine.get_text_utils()

                if isinstance(self.condition[u'keyword'], list):
                    etalons = self.condition[u'keyword']
                else:
                    etalons = [self.condition[u'keyword']]

                matchers = [KeywordMatcher.from_string(etalon) for etalon in etalons]

                for matcher in matchers:
                    if matcher.match(interpreted_phrase):
                        return True

                return False
        elif u'regex' in self.condition:
            rx_str = self.condition['regex']
            return re.search(rx_str, interpreted_phrase.interpretation, flags=re.IGNORECASE)
        else:
            raise NotImplementedError()

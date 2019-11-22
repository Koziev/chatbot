# -*- coding: utf-8 -*-

import logging
import re

from ruchatbot.bot.keyword_matcher import KeywordMatcher
from ruchatbot.utils.constant_replacer import replace_constant


class BaseRuleCondition(object):
    def __init__(self, data_yaml):
        self.key = u';'.join(k+'|'+str(v) for (k, v) in data_yaml.items())

    def get_short_repr(self):
        return None

    @staticmethod
    def from_yaml(condition_yaml, constants, text_utils):
        if u'intent' in condition_yaml:
            return RuleCondition_Intent(condition_yaml)
        elif u'state' in condition_yaml:
            return RuleCondition_State(condition_yaml)
        elif u'text' in condition_yaml:
            return RuleCondition_Text(condition_yaml, constants, text_utils)
        elif u'raw_text' in condition_yaml:
            return RuleCondition_RawText(condition_yaml, constants, text_utils)
        elif u'prev_bot_text' in condition_yaml:
            return RuleCondition_PrevBotText(condition_yaml, constants, text_utils)
        elif u'keyword' in condition_yaml:
            return RuleCondition_Keyword(condition_yaml, constants, text_utils)
        elif u'regex' in condition_yaml:
            return RuleCondition_Regex(condition_yaml, constants, text_utils)
        else:
            raise NotImplementedError()

        return res

    def get_key(self):
        return self.key

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
        raise NotImplementedError()


class RuleCondition_Intent(BaseRuleCondition):
    def __init__(self, data_yaml):
        super().__init__(data_yaml)
        self.intent = data_yaml[u'intent']

    def get_short_repr(self):
        return 'intent={}'.format(self.intent)

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        return self.intent == interpreted_phrase.intent


class RuleCondition_State(BaseRuleCondition):
    def __init__(self, data_yaml):
        super().__init__(data_yaml)
        self.slot_name = data_yaml['state']['slot']
        self.slot_mask = data_yaml['state']['regex_mask']

    def get_short_repr(self):
        return 'state {}={}'.format(self.slot_name, self.slot_mask)

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        slot_value = session.get_slot(self.slot_name)
        return slot_value and re.match(self.slot_mask, self.slot_value)


class RuleCondition_Text(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)
        if isinstance(data_yaml[u'text'], list):
            etalons = data_yaml[u'text']
        else:
            etalons = [data_yaml[u'text']]

        self.etalons = []
        for e in etalons:
            self.etalons.append(replace_constant(e, constants, text_utils))

    def get_short_repr(self):
        return 'text etalons[0/{}]="{}"'.format(len(self.etalons), self.etalons[0])

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        input_text = interpreted_phrase.interpretation
        return self.check_text(input_text, self.etalons, bot, session, interlocutor, interpreted_phrase, answering_engine)


class RuleCondition_RawText(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)
        if isinstance(data_yaml[u'raw_text'], list):
            etalons = data_yaml[u'raw_text']
        else:
            etalons = [data_yaml[u'raw_text']]

        self.etalons = []
        for e in etalons:
            self.etalons.append(replace_constant(e, constants, text_utils))

    def get_short_repr(self):
        return 'raw_text etalons[0/{}]="{}"'.format(len(self.etalons), self.etalons[0])

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        input_text = interpreted_phrase.raw_phrase
        return self.check_text(input_text, self.etalons, bot, session, interlocutor, interpreted_phrase, answering_engine)


class RuleCondition_PrevBotText(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)

        # Проверяем, что последняя фраза бота была похожа на указанный эталон
        if isinstance(data_yaml[u'prev_bot_text'], list):
            etalons = data_yaml[u'prev_bot_text']
        else:
            etalons = [data_yaml[u'prev_bot_text']]

        self.etalons = []
        for e in etalons:
            self.etalons.append(replace_constant(e, constants, text_utils))

    def get_short_repr(self):
        return 'prev_bot_text etalons[0/{}]="{}"'.format(len(self.etalons), self.etalons[0])

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        input_text = session.get_last_bot_utterance().interpretation
        return self.check_text(input_text, self.etalons, bot, session, interlocutor, interpreted_phrase,
                               answering_engine)


class RuleCondition_Keyword(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)

        if isinstance(data_yaml[u'keyword'], list):
            etalons = data_yaml[u'keyword']
        else:
            etalons = [data_yaml[u'keyword']]

        self.matchers = [KeywordMatcher.from_string(replace_constant(etalon, constants, text_utils))
                                                    for etalon
                                                    in etalons]

    def get_short_repr(self):
        return 'keyword matchers={}'.format(' '.join(map(str, self.matchers)))

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        for matcher in self.matchers:
            if not matcher.match(interpreted_phrase):
                return False

        return True


class RuleCondition_Regex(BaseRuleCondition):
    def __init__(self, data_yaml, constants, text_utils):
        super().__init__(data_yaml)
        self.rx_str = replace_constant(data_yaml['regex'], constants, text_utils)

    def get_short_repr(self):
        return 'regex="{}"'.format(self.rx_str)

    def check_condition(self, bot, session, interlocutor, interpreted_phrase, answering_engine):
        return re.search(self.rx_str, interpreted_phrase.interpretation, flags=re.IGNORECASE)

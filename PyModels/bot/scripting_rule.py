# -*- coding: utf-8 -*-

import random
import logging
#import parser
import yaml
import io
import pickle

from bot.interpreted_phrase import InterpretedPhrase
from bot.smalltalk_rules import SmalltalkSayingRule
from bot.smalltalk_rules import SmalltalkGeneratorRule
from generative_grammar.generative_grammar_engine import GenerativeGrammarEngine
from bot.comprehension_table import ComprehensionTable


class ScriptingRule(object):
    def __init__(self, condition, action):
        self.condition = condition
        self.action = action

    def check_condition(self, interpreted_phrase, answering_engine):
        if u'intent' in self.condition:
            return self.condition[u'intent'] == interpreted_phrase.intent
        elif u'text' in self.condition:
            if isinstance(self.condition[u'text'], list):
                text_utils = answering_engine.get_text_utils()
                word_embeddings = answering_engine.get_word_embeddings()

                # TODO: Эти приготовления будут выполняться при каждом запуске правила, надо бы
                # вынести их в конструктор правила.
                etalons = self.condition[u'text']
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
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def do_action(self, bot, session, user_id, interpreted_phrase):
        """Вернет True, если правило сформировало ответную реплику."""
        if u'say' in self.action:
            if isinstance(self.action[u'say'], list):
                bot.say(session, random.choice(self.action[u'say']))
            else:
                bot.say(session, self.action[u'say'])
            return True
        elif u'answer' in self.action:
            if isinstance(self.action[u'answer'], list):
                bot.push_phrase(user_id, random.choice(self.action[u'answer']), True)
            else:
                bot.push_phrase(user_id, self.action[u'answer'], True)
            return True
        elif u'callback' in self.action:
            resp = bot.invoke_callback(self.action[u'callback'], session, user_id, interpreted_phrase)
            if resp:
                bot.say(session, resp)
            return True
        elif u'nothing' in self.action:
            return False
        else:
            raise NotImplementedError()

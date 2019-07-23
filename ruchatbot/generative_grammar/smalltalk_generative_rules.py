# -*- coding: utf-8 -*-

from __future__ import print_function

import io
import yaml
import pickle

from ruchatbot.generative_grammar.generative_grammar_engine import GenerativeGrammarEngine
import ruchatbot.generative_grammar.questions_grammar_rules



class SmalltalkGenerativeRules(object):
    def __init__(self):
        self.rules = []

    @staticmethod
    def __get_node_list(node):
        if isinstance(node, list):
            return node
        else:
            return [node]

    @staticmethod
    def compile_yaml(yaml_filepath, output_filepath, gg_dictionaries):
        smalltalk_rule2grammar = dict()

        with io.open(yaml_filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

            for rule in data['smalltalk_rules']:
                condition = rule['rule']['if']
                action = rule['rule']['then']

                # Простые правила, которые задают срабатывание по тексту фразы, добавляем в отдельный
                # список, чтобы обрабатывать в модели синонимичности одним пакетом.
                if 'generate' in action:
                    condition_keyword = None
                    if 'text' in condition:
                        condition_keyword = u'text'
                    elif 'intent' in condition:
                        condition_keyword = u'intent'
                    else:
                        raise NotImplementedError()

                    for condition1 in SmalltalkGenerativeRules.__get_node_list(condition[condition_keyword]):
                        key = condition_keyword + u'|' + condition1
                        print(u'Rule "{}"...'.format(key))
                        generative_templates = list(SmalltalkGenerativeRules.__get_node_list(action['generate']))
                        grammar = GenerativeGrammarEngine()
                        grammar.set_dictionaries(gg_dictionaries)
                        ruchatbot.generative_grammar.questions_grammar_rules.compile_grammar(grammar, max_len=8, include_templates=False)

                        for template in generative_templates:
                            grammar.add_rule(template)

                        grammar.compile_rules()
                        smalltalk_rule2grammar[key] = grammar

        with open(output_filepath, 'wb') as f:
            pickle.dump(len(smalltalk_rule2grammar), f)
            for key, grammar in smalltalk_rule2grammar.items():
                pickle.dump(key, f)
                grammar.pickle_to(f)

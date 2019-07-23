# -*- coding: utf-8 -*-
"""
Сборка правил и прочил вспомогательных структур для работы генеративной грамматики
в модуле генерации ответа и в smalltalk-правилах.

Сохранение результатов на диск для последующего использования чатботом.
"""

from __future__ import print_function

import os
import gc
import io

from ruchatbot.generative_grammar.answer_generator_engine import AnswerGeneratorEngine
from ruchatbot.generative_grammar.generative_grammar_engine import GenerativeGrammarDictionaries
from ruchatbot.generative_grammar.smalltalk_generative_rules import SmalltalkGenerativeRules
from ruchatbot.generative_grammar.generative_grammar_engine import GenerativeGrammarEngine
import ruchatbot.generative_grammar.questions_grammar_rules


if __name__ == '__main__':
    model_folder = '../../../tmp'
    tmp_folder = '../../../tmp'
    data_folder = '../../../data'

    # Список слов, которые упоминаются в датасетах
    known_words = set()
    with io.open(os.path.join(tmp_folder, 'dataset_words.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            known_words.add(line.strip())

    # Словари общие для нескольких грамматик
    print('Build dictionaries...')
    dictionaries = GenerativeGrammarDictionaries()
    dictionaries.prepare(data_folder, max_ngram_gap=1, use_assocs=False, lexicon_words=known_words,
                         use_verb_prep_case=True)
    dictionaries.save(os.path.join(model_folder, 'generative_grammar_dictionaries.bin'))

    # Теперь генератор ответов
    print('Build answer grammar...')
    answer_generator = AnswerGeneratorEngine()
    answer_generator.set_dictionaries(dictionaries)
    answer_generator.compile_grammar(data_folder, tmp_folder)
    del answer_generator
    gc.collect()

    # Генеративные грамматики для smalltalk-правил
    print('Compile rules.yaml')
    SmalltalkGenerativeRules.compile_yaml(os.path.join(data_folder, 'rules.yaml'),
                                          os.path.join(tmp_folder, 'smalltalk_generative_grammars.bin'),
                                          dictionaries)

    # Генератор реплик
    print('Build replica generator grammar...')
    grammar_path = os.path.join(tmp_folder, 'replica_generator_grammar.bin')
    grammar = GenerativeGrammarEngine()
    grammar.set_dictionaries(dictionaries)
    ruchatbot.generative_grammar.questions_grammar_rules.compile_grammar(grammar, max_len=8)
    grammar.save(grammar_path)
    del grammar
    gc.collect()


    print('Add done.')

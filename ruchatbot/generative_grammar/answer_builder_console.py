# -*- coding: utf-8 -*-
"""
Проверка набора моделей и генеративной грамматики на задаче генерации ответа на
вопрос при изместной предпосылке.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import time

from utils.tokenizer import Tokenizer
import utils.console_helpers
import utils.logging_helpers

from generative_grammar.generative_grammar_engine import GenerativeGrammarDictionaries
from generative_grammar.answer_generator_engine import AnswerGeneratorEngine
from generative_grammar.word_embeddings import WordEmbeddings


if __name__ == '__main__':
    model_folder = '../../tmp'
    tmp_folder = '../../tmp'
    data_folder = '../../data'

    word2vector_path = os.path.expanduser('~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin')

    dictionaries = GenerativeGrammarDictionaries()
    dict_path = os.path.join(tmp_folder, 'console_generator_dictionaries.bin')
    # Загружаем ранее собранные и сохраненные словари
    dictionaries.load(dict_path)


    generator = AnswerGeneratorEngine()
    generator.set_dictionaries(dictionaries)
    generator.load_models(model_folder)
    generator.verbosity = 2

    # По этому списку слов будет отсекать всякую экзотичную лексику
    generator.load_known_words(os.path.join(tmp_folder, 'dataset_words.txt'))

    word2vec = WordEmbeddings()
    word2vec.load_embeddings(model_folder, word2vector_path)

    tokenizer = Tokenizer()
    tokenizer.load()

    while True:
        premise = utils.console_helpers.input_kbd('premise:> ').strip().lower()
        premises = [premise]
        question = utils.console_helpers.input_kbd('question:> ').strip().lower()

        start_time = time.time()
        answer = generator.generate_answer(premises, question, tokenizer, word2vec)
        elapsed_time = time.time() - start_time
        print('{} sec elapsed'.format(elapsed_time))
        print(u'answer={} ({})'.format(answer.get_str(), answer.get_rank()))


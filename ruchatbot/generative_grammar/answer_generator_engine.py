# -*- coding: utf-8 -*-
"""
Движок генерации ответа на вопрос при известной предпосылке.
Объединяет несколько ранее обученных моделей.
"""

from __future__ import division
from __future__ import print_function

import itertools
import json
import os
import io
import numpy as np
import logging

from keras.models import model_from_json

from ruchatbot.generative_grammar import answers_grammar_rules
from ruchatbot.generative_grammar.generative_grammar_engine import GenerativeGrammarEngine
from ruchatbot.generative_grammar.word_selector import WordSelector
from ruchatbot.generative_grammar.answer_length_predictor import AnswerLengthPredictor
from ruchatbot.generative_grammar.answer_relevancy import AnswerRelevancy


PAD_WORD = u''
padding = 'left'


#def pad_wordseq(words, n):
#    """Слева добавляем пустые слова"""
#    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


#def rpad_wordseq(words, n):
#    """Справа добавляем пустые слова"""
#    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


bad_tokens = set(u'? . !'.split())



class AnswerGeneratorEngine(object):
    def __init__(self):
        self.grammar = None
        self.word_selector = None
        self.len_predictor = None
        self.answer_relevancy = None
        self.known_words = None  # set
        self.verbosity = 0

    def set_dictionaries(self, dictionaries):
        self.dictionaries = dictionaries

    def compile_grammar(self, data_folder, tmp_folder):
        self.grammar = GenerativeGrammarEngine()
        self.grammar.set_dictionaries(self.dictionaries)
        answers_grammar_rules.compile_grammar(self.grammar, max_len=5)
        self.grammar.save(os.path.join(tmp_folder, 'answer_generative_grammar.bin'))

    def load_models(self, model_folder):
        assert(self.dictionaries is not None)

        with open(os.path.join(model_folder, 'answer_generative_grammar.bin'), 'rb') as f:
            self.grammar = GenerativeGrammarEngine.unpickle_from(f)

        self.grammar.set_dictionaries(self.dictionaries)

        self.word_selector = WordSelector()
        self.word_selector.load(model_folder)

        self.len_predictor = AnswerLengthPredictor()
        self.len_predictor.load(model_folder)

        self.answer_relevancy = AnswerRelevancy()
        self.answer_relevancy.load(model_folder)

    def load_known_words(self, file_path):
        # По этому списку слов будет отсекать всякую экзотичную лексику
        self.known_words = set()
        with io.open(file_path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                word = line.strip()
                self.known_words.add(word)

    def get_w2v_path(self):
        return self.word_selector.get_w2v_path()

    def generate_answer(self, str_premises, str_question, tokenizer, word2vec):
        premises = [list(filter(lambda w: w not in bad_tokens, tokenizer.tokenize(premise.lower()))) for premise in str_premises]
        question = list(filter(lambda w: w not in bad_tokens, tokenizer.tokenize(str_question.lower())))

        word_p = self.word_selector.select_words(premises, question, word2vec)

        if self.verbosity >= 1:
            print('Selected words and their weights:')
            for word, p in sorted(word_p, key=lambda z: -z[1]):
                print(u'{:15s}\t{}'.format(word, p))

        len2proba = self.len_predictor.predict(premises, question, word2vec)
        # начало отладки
        max_len_p = 0.0
        best_len = 0
        for l, p in len2proba.items():
            if p > max_len_p:
                max_len_p = p
                best_len = l

        if self.verbosity >= 1:
            print(u'Most probable answer length={} (p={})'.format(best_len, max_len_p))
            #print('Answer length probabilities:')
            #for l, p in len_p.items():
            #    print('len={} p={}'.format(l, p))
            # конец отладки

        # Оптимизация от 20-05-2019:
        # 1) отбрасываем слишком малозначимые слова
        p_threshold = max(p for word, p in word_p) * 0.02
        word_p = [(word, p) for word, p in word_p if p > p_threshold]
        # 2) если слов все равно осталось много, то оставим максимальную длину + 1
        if len(word_p) > (best_len+1):
            word_p = sorted(word_p, key=lambda z: -z[1])[:best_len + 1]

        # НАЧАЛО ОТЛАДКИ
        #if str_premises[0] == u'аборигены окучивают картофан':
        #    print('DEBUG@126')
        # КОНЕЦ ОТЛАДКИ
        #print('Generating answers...')
        # todo - передавать доп. функцию взвешивания сгенерированных ответов, в том числе с
        # помощью модель nn_answer_relevancy
        all_generated_phrases = self.grammar.generate2(word_p, self.known_words)

        #print('Ranking the answers...')

        answers = self.answer_relevancy.score_answers(premises, question, all_generated_phrases, word2vec, tokenizer, len2proba)

        # начало отладки
        #for a in answers:
        #    if a.words[0] == u'дядя':
        #        print(u'DEBUG@340 {} {}'.format(a.get_proba0(), a.get_rank()))
        #        exit(0)
        # конец отладки

        sorted_answers = sorted(answers, key=lambda z: -z.get_rank())

        if self.verbosity >= 2:
            for phrase in sorted_answers[:100]:
                print(u'{:6f}\t{}'.format(phrase.get_rank(), phrase.get_str()))

        if len(sorted_answers) == 0:
            return None
        else:
            return sorted_answers[0]

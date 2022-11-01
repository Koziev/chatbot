"""
NLP Pipeline чатбота

Содержит код для выполнения операций с текстом на русском языке (или на другом целевом),
в частности - токенизация, лемматизация, частеречная разметка. Для этого грузит
разнообразные пакеты, включая POS Tagger, чанкер etc.
Различные словарные базы.

Для проекта чатбота https://github.com/Koziev/chatbot

01.12.2021 Добавляем UDPipe в пайплайн для реализации детектора гендерной самоидентификации собеседника
           и для задач аугментации.
05.05.2021 Грузим список имен, чтобы фильтровать результаты генерации читчата
25.04.2022 Большой рефакторинг и чистка кода в связи с переходом на новую архитектуру
"""

import re
import os
import logging
import pickle

import pyconll
from ufal.udpipe import Model, Pipeline, ProcessingError

import rutokenizer
import rupostagger
import ruword2tags

from ruchatbot.utils.tokenizer import Tokenizer
from ruchatbot.bot.language_resources import LanguageResources


class TextUtils(object):
    def __init__(self):
        self.clause_splitter = rutokenizer.Segmenter()
        self.tokenizer = Tokenizer()
        self.tokenizer.load()
        self.language_resources = LanguageResources()
        self.postagger = rupostagger.RuPosTagger()
        self.word2tags = ruword2tags.RuWord2Tags()
        self.names = None

    def load_dictionaries(self, data_folder, models_folder):
        self.postagger.load()

        self.word2tags.load()

        # Грузим dependency parser UDPipe и русскоязычную модель
        model_file = os.path.join(models_folder, 'udpipe_syntagrus.model')
        self.udpipe_model = Model.load(model_file)
        self.udpipe_pipeline = Pipeline(self.udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        self.udpipe_error = ProcessingError()

        with open(os.path.join(models_folder, 'names.pkl'), 'rb') as f:
            self.names = set(pickle.load(f).keys())

    def apply_word_function(self, func, constants, words):
        part_of_speech = None
        tag = None
        if func == '$chooseAdjByGender':
            part_of_speech = 'ПРИЛАГАТЕЛЬНОЕ'
            tag = ('РОД', constants['gender'])
        elif func == '$chooseVByGender':
            part_of_speech = 'ГЛАГОЛ'
            tag = ('РОД', constants['gender'])
        elif func == '$chooseNByGender':
            part_of_speech = 'СУЩЕСТВИТЕЛЬНОЕ'
            tag = ('РОД', constants['gender'])
        else:
            raise NotImplementedError()

        tag2 = tag[0] + '=' + tag[1]
        for word in words:
            for tagset in self.word2tags[word.lower()]:
                if part_of_speech in tagset and tag2 in tagset:
                    return word

        msg = 'Could not choose a word among {}'.format(' '.join(words))
        raise RuntimeError(msg)

    def tag(self, words):
        """ Частеречная разметка для цепочки слов words """
        return self.postagger.tag(words)

    def canonize_text(self, s):
        """ Удаляем два и более пробелов подряд, заменяя на один """
        s = re.sub("(\\s{2,})", ' ', s.strip())
        return s

    def remove_terminators(self, s):
        """ Убираем финальные пунктуаторы ! ? ."""
        return s[:-1].strip() if s[-1] in '?!.' else s

    def normalize_delimiters(self, s):
        return s.replace(' ?', '?').replace(' ,', ',').replace(' .', '.').replace(' !', '!')

    def wordize_text(self, s):
        return self.normalize_delimiters(' '.join(self.tokenize(s)))

    def ngrams(self, s, n):
        return [''.join(z) for z in zip(*[s[i:] for i in range(n)])]

    def split_clauses(self, s):
        return list(self.clause_splitter.split(s))

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def is_question_word(self, word):
        return word in 'насколько где кто что почему откуда куда зачем чего кого кем чем кому чему ком чем как сколько ли когда докуда какой какая какое какие какого какую каких каким какими какому какой каков какова каковы'.split()

    def build_output_phrase(self, words):
        s = ' '.join(words)
        s = s.replace(u' ?', u'?').replace(u' !', u'!').replace(u' ,', u',').replace(u' :', u',') \
            .replace(u' .', u'.').replace(u'( ', u'(').replace(u' )', u')')
        s = s[0].upper() + s[1:]
        return s

    def detect_person0(self, words):
        if any((word in (u'ты', u'тебя', u'тебе')) for word in words):
            return 2

        if any((word in (u'я', u'мне', u'меня')) for word in words):
            return 1

        return -1

    def parse_syntax(self, text_str):
        processed = self.udpipe_pipeline.process(text_str, self.udpipe_error)
        if self.udpipe_error.occurred():
            logging.error("An error occurred when running run_udpipe: %s", self.udpipe_error.message)
            return None

        parsed_data = pyconll.load_from_string(processed)[0]
        return parsed_data

    def get_udpipe_attr(self, token, tag_name):
        if tag_name in token.feats:
            v = list(token.feats[tag_name])[0]
            return v

        return ''

    def contains_name(self, text_str) -> bool:
        parsed_data = self.parse_syntax(text_str)

        up_words = [z.form.lower().replace('ё', 'е') for z in parsed_data]
        up_lemmas = [z.lemma.lower().replace('ё', 'е') for z in parsed_data]

        return any((w in self.names) for w in up_words) or any((l in self.names) for l in up_lemmas)

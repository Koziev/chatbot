# -*- coding: utf-8 -*-
"""
Код для выполнения операций с текстом на русском языке (или другом целевом),
в частности - токенизация, лемматизация, частеречная разметка.
Загрузка различных словарных баз.
"""

import itertools
import re
import os
import io
import yaml

from pymystem3 import Mystem
import rupostagger
from utils.tokenizer import Tokenizer
from bot.word2lemmas import Word2Lemmas
from bot.language_resources import LanguageResources
from generative_grammar.generative_grammar_engine import GenerativeGrammarDictionaries


BEG_WORD = u'\b'
END_WORD = u'\n'
PAD_WORD = u''


class TextUtils(object):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.load()
        self.lemmatizer = Mystem()
        self.lexicon = Word2Lemmas()
        self.language_resources = LanguageResources()
        self.postagger = rupostagger.RuPosTagger()
        self.gg_dictionaries = GenerativeGrammarDictionaries()
        self.known_words = set()

    def load_dictionaries(self, data_folder, models_folder):
        # Общий словарь для генеративных грамматик
        self.gg_dictionaries.load(os.path.join(models_folder, 'generative_grammar_dictionaries.bin'))

        word2lemmas_path = os.path.join(data_folder, 'ru_word2lemma.tsv.gz')
        self.lexicon.load(word2lemmas_path)

        word2tags_path = os.path.join(data_folder, 'chatbot_word2tags.dat')
        self.postagger.load(word2tags_path)

        rules_path = os.path.join(data_folder, 'rules.yaml')
        with io.open(rules_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.no_info_replicas = data['no_relevant_information']
            self.unknown_order = data['unknown_order']

            self.language_resources.key2phrase[u'yes'] = data[u'answers'][u'yes']
            self.language_resources.key2phrase[u'not'] = data[u'answers'][u'not']

        # Список "хороших слов" для генеративной грамматики
        with io.open(os.path.join(models_folder, 'dataset_words.txt'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                word = line.strip()
                self.known_words.add(word)



    def tag(self, words):
        """ Частеречная разметка для цепочки слов words """
        return self.postagger.tag(words)

    def canonize_text(self, s):
        """ Удаляем два и более пробелов подряд, заменяя на один """
        s = re.sub("(\\s{2,})", ' ', s.strip())
        return s

    def remove_terminators(self, s):
        """ Убираем финальные пунктуаторы ! ? ."""
        return s[:-1].strip() if s[-1] in u'?!.' else s

    def wordize_text(self, s):
        return u' '.join(self.tokenize(s))

    def ngrams(self, s, n):
        return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]

    def words2str(self, words):
        return u' '.join(itertools.chain([BEG_WORD], filter(lambda z: len(z) > 0, words), [END_WORD]))

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def lemmatize(self, s):
        words = self.tokenizer.tokenize(s)
        wx = u' '.join(words)
        return [l for l in self.lemmatizer.lemmatize(wx) if len(l.strip()) > 0]

    def lpad_wordseq(self, words, n):
        """ Слева добавляем пустые слова """
        return list(itertools.chain(itertools.repeat(PAD_WORD, n - len(words)), words))

    def rpad_wordseq(self, words, n):
        """ Справа добавляем пустые слова """
        return list(itertools.chain(words, itertools.repeat(PAD_WORD, n - len(words))))

    def get_lexicon(self):
        return self.lexicon

    def is_question_word(self, word):
        return word in u'кто что почему откуда куда зачем чего кого кем чем кому чему ком чем как сколько ли когда докуда какой какая какое какие какого какую каких каким какими какому какой'.split()

    def build_output_phrase(self, words):
        s = u' '.join(words)
        s = s.replace(u' ?', u'?').replace(u' !', u'!').replace(u' ,', u',').replace(u' :', u',') \
            .replace(u' .', u'.').replace(u'( ', u'(').replace(u' )', u')')
        s = s[0].upper() + s[1:]
        return s

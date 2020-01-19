# -*- coding: utf-8 -*-
"""
NLP Pipeline чатбота

Содержит код для выполнения операций с текстом на русском языке (или на другом целевом),
в частности - токенизация, лемматизация, частеречная разметка. Для этого грузит
разнообразные пакеты, включая POS Tagger, чанкер etc.
Различные словарные базы.

Для проекта чатбота https://github.com/Koziev/chatbot
"""

import itertools
import re
import os
import io
import yaml

import rupostagger
import rulemma
import ruchunker
#import rusyntax2
import ruword2tags

from ruchatbot.utils.tokenizer import Tokenizer
from ruchatbot.bot.word2lemmas import Word2Lemmas
from ruchatbot.bot.language_resources import LanguageResources
from ruchatbot.generative_grammar.generative_grammar_engine import GenerativeGrammarDictionaries
from ruchatbot.bot.word_embeddings import WordEmbeddings
from ruchatbot.bot.string_constants import BEG_WORD, END_WORD, PAD_WORD


class PhraseToken:
    def __init__(self):
        self.word = None
        self.norm_word = None
        self.lemma = None
        self.word_index = None
        self.chunk_index = None
        self.tagset = None
        self.is_chunk_starter = None

    def __repr__(self):
        return self.word


class TextUtils(object):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.load()
        self.lexicon = Word2Lemmas()
        self.language_resources = LanguageResources()
        self.postagger = rupostagger.RuPosTagger()
        self.chunker = ruchunker.Chunker()
        self.word2tags = ruword2tags.RuWord2Tags()
        self.flexer = ruword2tags.RuFlexer()
        self.syntan = None
        self.gg_dictionaries = GenerativeGrammarDictionaries()
        self.known_words = set()
        #self.lemmatizer = Mystem()
        self.lemmatizer = rulemma.Lemmatizer()
        self.word_embeddings = None

    def load_embeddings(self, w2v_dir, wc2v_dir):
        # Загрузка векторных словарей
        self.word_embeddings = WordEmbeddings()
        self.word_embeddings.load_models(w2v_dir)

        p = os.path.join(wc2v_dir, 'wc2v.kv')
        self.word_embeddings.load_wc2v_model(p)

        p = os.path.join(w2v_dir, 'w2v.kv')
        self.word_embeddings.load_w2v_model(p)

    def load_dictionaries(self, data_folder, models_folder):
        self.lemmatizer.load()

        # Общий словарь для генеративных грамматик
        self.gg_dictionaries.load(os.path.join(models_folder, 'generative_grammar_dictionaries.bin'))

        word2lemmas_path = os.path.join(data_folder, 'ru_word2lemma.tsv.gz')
        self.lexicon.load(word2lemmas_path)

        #word2tags_path = os.path.join(data_folder, 'chatbot_word2tags.dat')
        #self.postagger.load(word2tags_path)
        self.postagger.load()

        self.word2tags.load()
        self.flexer.load()
        self.chunker.load()

        #self.syntan = rusyntax2.Tagger(self.word2tags, w2v, self.postagger)
        #self.syntan.load()

        rules_path = os.path.join(data_folder, 'rules.yaml')
        with io.open(rules_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            #self.no_info_replicas = data['no_relevant_information']
            #self.unknown_order = data['unknown_order']

            self.language_resources.key2phrase[u'yes'] = data[u'answers'][u'yes']
            self.language_resources.key2phrase[u'not'] = data[u'answers'][u'not']

        # Список "хороших слов" для генеративной грамматики
        with io.open(os.path.join(models_folder, 'dataset_words.txt'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                word = line.strip()
                self.known_words.add(word)

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

        for word in words:
            tagsets = self.gg_dictionaries.grdict.get_word_tagsets2(word.lower(), part_of_speech)
            if any((tag in tagset) for tagset in tagsets):
                return word

        msg = u'Could not choose a word among {}'.format(u' '.join(words))
        raise RuntimeError(msg)

    def tag(self, words, with_lemmas=False):
        """ Частеречная разметка для цепочки слов words """
        if with_lemmas:
            return self.lemmatizer.lemmatize(self.postagger.tag(words))
        else:
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
        #return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]
        return [u''.join(z) for z in zip(*[s[i:] for i in range(n)])]

    def words2str(self, words):
        return u' '.join(itertools.chain([BEG_WORD], filter(lambda z: len(z) > 0, words), [END_WORD]))

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def extract_lemma(self, token):
        return token[0] if token[1] == 'PRON' else token[2]

    def lemmatize(self, s):
        words = self.tokenizer.tokenize(s)
        #wx = u' '.join(words)
        #return [l for l in self.lemmatizer.lemmatize(wx) if len(l.strip()) > 0]
        tokens = self.lemmatizer.lemmatize(self.postagger.tag(words))
        return [self.extract_lemma(t) for t in tokens]

    def lpad_wordseq(self, words, n):
        """ Слева добавляем пустые слова """
        return list(itertools.chain(itertools.repeat(PAD_WORD, n - len(words)), words))

    def rpad_wordseq(self, words, n):
        """ Справа добавляем пустые слова """
        return list(itertools.chain(words, itertools.repeat(PAD_WORD, n - len(words))))

    def get_lexicon(self):
        return self.lexicon

    def is_question_word(self, word):
        return word in u'насколько где кто что почему откуда куда зачем чего кого кем чем кому чему ком чем как сколько ли когда докуда какой какая какое какие какого какую каких каким какими какому какой'.split()

    def build_output_phrase(self, words):
        s = u' '.join(words)
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

    def extract_chunks(self, sample):
        tokens = self.tokenizer.tokenize(sample)
        tagsets = list(self.postagger.tag(tokens))
        lemmas = self.lemmatizer.lemmatize(tagsets)
        #edges = syntan.parse(tokens, tagsets)

        phrase_tokens = []
        for word_index, (token, tagset, lemma) in enumerate(zip(tokens, tagsets, lemmas)):
            t = PhraseToken()
            t.word = token
            t.norm_word = token.lower()
            t.lemma = lemma[2]
            t.tagset = tagset[1]
            t.word_index = word_index
            phrase_tokens.append(t)

        chunks = self.chunker.parse(tokens)
        for chunk_index, chunk in enumerate(chunks):
            phrase_tokens[chunk.tokens[0].index].is_chunk_starter = True
            for token in chunk.tokens:
                phrase_tokens[token.index].chunk_index = chunk_index

        return chunks

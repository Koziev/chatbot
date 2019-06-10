# -*- coding: utf-8 -*-
"""
Движок генеративной грамматики для вопросно-ответной системы.
"""

from __future__ import division
from __future__ import print_function

import math
import os
import pickle
import codecs
import io
import itertools
import collections
import operator
import random
import six
import logging
import re

import rutokenizer


def decode_pos(pos):
    if pos in [u'ДЕЕПРИЧАСТИЕ', u'ГЛАГОЛ', u'ИНФИНИТИВ']:
        # Объединяем глагольные части речи.
        return u'ГЛАГОЛ'
    else:
        return pos


def clean_output(s):
    s = s.replace(u' ?', u'?').replace(u' !', u'!').replace(u' ,', u',').replace(u' :', u',')\
        .replace(u' .', u'.').replace(u'( ', u'(').replace(u' )', u')')
    s = s[0].upper()+s[1:]
    return s


class CorpusWords:
    """ Анализ большого текстового корпуса и сбор самых частотных слов """
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.storage_path = '../../tmp/all_words.dat'
        self.all_words = set()

    def analyze(self, corpora, thesaurus_path, grdict_path, word2freq_wiki_path):
        #tokenizer = rutokenizer.Tokenizer()
        #tokenizer.load()

        self.all_words = set()  # здесь накопим слова, которые будут участвовать в перефразировках.

        # Тезаурус содержит связи между леммами, соберем список этих лемм.
        thesaurus_entries = set()
        with codecs.open(thesaurus_path, 'r', 'utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('|')
                if len(tx) == 5:
                    word1 = tx[0].replace(u' - ', u'-')
                    pos1 = decode_pos(tx[1])
                    word2 = tx[2].replace(u' - ', u'-')
                    pos2 = decode_pos(tx[3])

                    thesaurus_entries.add((word1, pos1))
                    thesaurus_entries.add((word2, pos2))

        # Теперь для всех лемм, упомянутых в тезаурусе, получим все грамматические формы.
        thesaurus_forms = set()
        with codecs.open(grdict_path, 'r', 'utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 4:
                    word = tx[0].replace(u' - ', u'-')
                    pos = decode_pos(tx[1])
                    lemma = tx[2]

                    if (lemma, pos) in thesaurus_entries:
                        thesaurus_forms.add(word)

        self.all_words.update(thesaurus_forms)

        if True:
            # добавим слова из ассоциаций
            with codecs.open(word2freq_wiki_path, 'r', 'utf-8') as rdr:
                for line in rdr:
                    word = line.strip()
                    self.all_words.add(word)

        if False:
            # Теперь читаем текстовый корпус.
            self.word2freq = collections.Counter()
            for fname in corpora:
                logging.info(u'Reading corpus from "{}"'.format(fname))
                with codecs.open(fname, 'r', 'utf-8') as rdr:
                    for iline, line in enumerate(rdr):
                        phrase = line.strip()
                        words = tokenizer.tokenize(phrase)
                        self.word2freq.update(words)
                        #phrase = u' '.join(words)
                        # tfidf_corpus.add(phrase)
                        if iline > 2000000:
                            break

            self.all_words.update(w for (w, freq) in self.word2freq.items() if freq > self.min_freq)
            #logging.info('{} words with more than {} occurencies in corpus vocabulary'.format(len(self.all_words), self.min_freq))

        logging.info('Total number of words in corpus={}'.format(len(self.all_words)))

    def __contains__(self, item):
        return item in self.all_words

    def save(self):
        logging.info(u'Storing frequent words in {}'.format(self.storage_path))
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.all_words, f)

        if True:
            with codecs.open('../../tmp/all_words.txt', 'w', 'utf-8') as wrt:
                for word in self.all_words:
                    wrt.write(u'{}\n'.format(word))

    def load(self):
        logging.info(u'Loading frequent words from {}'.format(self.storage_path))
        with open(self.storage_path, 'rb') as f:
            self.all_words = pickle.load(f)

    def list(self):
        return self.all_words


class Associations:
    def __init__(self):
        self.word2assocs = dict()

    def load(self, path, grdict):
        logging.info(u'Loading associations from {}'.format(path))

        word2assocs = dict()
        with codecs.open(path, 'r', 'utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) >= 3:
                    word1 = tx[0]
                    word2 = tx[1]
                    mi = float(tx[2].replace(',', '.'))

                    #if word1 in grdict and word2 in grdict\
                    #        and not stop_words.is_stop_word2(word1)\
                    #        and not stop_words.is_stop_word2(word2):
                    if True:
                        if word1 not in word2assocs:
                            word2assocs[word1] = [(word2, mi)]
                        else:
                            word2assocs[word1].append((word2, mi))

                        #if word2 not in word2assocs:
                        #    word2assocs[word2] = [(word1, mi)]
                        #else:
                        #    word2assocs[word2].append((word1, mi))


        # оставим для каждого слова не более nbest ассоциаций
        nbest = 50
        self.word2assocs = dict()
        for word, assocs in six.iteritems(word2assocs):
            if len(assocs) > nbest:
                assocs = sorted(assocs, key=operator.itemgetter(1), reverse=True)[:nbest]
            self.word2assocs[word] = assocs

    def collect_from_corpus(self, corpora, max_pairs):
        word2freq = collections.Counter()
        pair2freq = dict()
        for corpus_path in corpora:
            logging.info(u'Collecting ngram statistics from {}'.format(corpus_path))
            if len(pair2freq) > max_pairs:
                break

            with io.open(corpus_path, 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    if len(pair2freq) > max_pairs:
                        break

                    s = line.strip().replace(u'Q: ', u'').replace(u'A: ', u'').replace(u'T: ', u'')
                    words = s.split()
                    word2freq.update(words)
                    nword = len(words)
                    for i1, word1 in enumerate(words):
                        for i2 in range(i1+1, nword):
                            word2 = words[i2]
                            w = 1.0/(i2-i1)
                            k = (word1, word2)
                            pair2freq[k] = pair2freq.get(k, 0.0) + w

        N1 = float(sum(word2freq.values()))  # общая частота всех слов
        N2 = sum(pair2freq.values())  # общая частота всех пар
        word2assocs = dict()

        for pair, n2 in six.iteritems(pair2freq):
            a = n2 / float(N2)
            word1 = pair[0]
            word2 = pair[1]
            f1 = word2freq[word1] / N1
            f2 = word2freq[word2] / N1
            mi = a * math.log(a / (f1 * f2))
            if word1 not in word2assocs:
                word2assocs[word1] = [(word2, mi)]
            else:
                word2assocs[word1].append((word2, mi))

        # оставим для каждого слова не более nbest ассоциаций
        nbest = 50
        for word, assocs in six.iteritems(word2assocs):
            if len(assocs) > nbest:
                assocs = sorted(assocs, key=operator.itemgetter(1), reverse=True)[:nbest]
            self.word2assocs[word] = assocs

        # НАЧАЛО ОТЛАДКИ
        #ax = self.word2assocs[u'ловит']
        #for word2, mi in ax:
        #    print(u'{}={}'.format(word2, mi))
        #exit(0)
        # КОНЕЦ ОТЛАДКИ


    def get_assocs(self, word):
        return self.word2assocs.get(word, [])

    def get_mi(self, word1, word2):
        if word1 in self.word2assocs:
            for w2, mi in self.word2assocs[word1]:
                if w2 == word2:
                    return mi

        return 0.0


class NGrams(object):
    def __init__(self):
        self.all_2grams = None
        self.all_3grams = None
        self.has_2grams = False
        self.has_3grams = False

    def collect(self, corpora, max_gap, max_2grams, max_3grams):
        tokenizer = rutokenizer.Tokenizer()
        tokenizer.load()

        self.all_2grams = collections.Counter()
        self.all_3grams = collections.Counter()

        for corpus_path in corpora:
            logging.info(u'Collecting ngram statistics from {}'.format(corpus_path))

            if len(self.all_2grams) > max_2grams and len(self.all_3grams) > max_3grams:
                break

            with io.open(corpus_path, 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    if line.startswith(u'#'):
                        continue

                    s = line.strip().replace(u'Q: ', u'').replace(u'A: ', u'').replace(u'T: ', u'')
                    s = s.replace(u'(+)', u'').replace(u'(-)', u'').strip()

                    s2 = s.split(u'|')
                    for s in s2:
                        words = tokenizer.tokenize(s.lower())
                        for gap in range(max_gap):
                            ngrams = list(zip(words, words[1+gap:]))
                            self.all_2grams.update(ngrams)

                            ngrams = list(zip(words, words[1+gap:], words[2+gap:]))
                            self.all_3grams.update(ngrams)

                    if len(self.all_2grams) > max_2grams and len(self.all_3grams) > max_3grams:
                        break

        self.has_2grams = len(self.all_2grams) > 0
        self.has_3grams = len(self.all_3grams) > 0
        logging.info('{} 2grams, {} 3grams stored'.format(len(self.all_2grams), len(self.all_3grams)))

    def __contains__(self, ngram):
        l = len(ngram)
        if l == 2:
            return ngram in self.all_2grams
        elif l == 3:
            return ngram in self.all_3grams
        else:
            return False

    def get(self, ngram):
        l = len(ngram)
        if l == 2:
            return self.all_2grams.get(ngram, 0)
        elif l == 3:
            return self.all_3grams.get(ngram, 0)
        else:
            return False


class Word2Lemmas(object):
    def __init__(self):
        self.lemmas = dict()
        self.forms = dict()  # для каждой формы - список лемм с пометками части речи
        self.verbs = set()
        self.nouns = set()
        self.adjs = set()
        self.advs = set()

    def load(self, path, all_words):
        logging.info(u'Loading lexicon from {}'.format(path))
        with open(path, 'r') as rdr:
            for line in rdr:
                tx = line.strip().decode('utf8').split('\t')
                if len(tx) == 4:
                    if int(tx[3]) >= 0:
                        form = tx[0].replace(u' - ', u'-').lower()
                        if all_words is None or form in all_words:
                            lemma = tx[1].replace(u' - ', u'-').lower()
                            pos = decode_pos(tx[2])

                            if form not in self.forms:
                                self.forms[form] = [(lemma, pos)]
                            else:
                                self.forms[form].append((lemma, pos))

                            k = lemma+'|'+pos
                            if k not in self.lemmas:
                                self.lemmas[k] = {form}
                            else:
                                self.lemmas[k].add(form)

                            if pos == u'ГЛАГОЛ':
                                self.verbs.add(lemma)
                            elif pos == u'СУЩЕСТВИТЕЛЬНОЕ':
                                self.nouns.add(lemma)
                            elif pos == u'ПРИЛАГАТЕЛЬНОЕ':
                                self.adjs.add(lemma)
                            elif pos == u'НАРЕЧИЕ':
                                self.advs.add(lemma)

        logging.info('Lexicon loaded: {} lemmas, {} wordforms'.format(len(self.lemmas), len(self.forms)))

    def same_lemma(self, word1, word2):
        return len(set(self.forms.get(word1, [])) & set(self.forms.get(word2, []))) > 1

    def get_lemma(self, word):
        if word in self.forms:
            return self.forms[word][0][0]
        else:
            return word

    def get_lemmas(self, word):
        if word in self.forms:
            return self.forms[word]
        else:
            return [(word, None)]

    def get_forms(self, word, part_of_speech):
        if word in self.forms:
            forms = set()
            for lemma, pos in self.forms[word]:
                if lemma is None or part_of_speech is None:
                    raise RuntimeError()

                if pos == part_of_speech:
                    k = lemma+u'|'+part_of_speech
                    if k in self.lemmas:
                        forms.update(self.lemmas[k])
            return forms
        else:
            k = word+ u'|' + part_of_speech
            if k in self.lemmas:
                return list(self.lemmas[k])

            return [word]

    def get_random_verb(self, valid_words):
        return random.choice([w for w in self.verbs if w in valid_words])

    def get_random_noun(self, valid_words):
        return random.choice([w for w in self.nouns if w in valid_words])

    def get_random_adj(self, valid_words):
        return random.choice([w for w in self.adjs if w in valid_words])

    def get_random_adv(self, valid_words):
        return random.choice([w for w in self.advs if w in valid_words])


class Thesaurus:
    def __init__(self):
        self.word2links = dict()

    def load(self, thesaurus_path):
        logging.info(u'Loading thesaurus from "{}"'.format(thesaurus_path))
        with codecs.open(thesaurus_path, 'r', 'utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 5:
                    word1 = tx[0].replace(u' - ', u'-').lower()
                    pos1 = tx[1]
                    word2 = tx[2].replace(u' - ', u'-').lower()
                    pos2 = tx[3]
                    relat = tx[4]

                    if relat in (u'в_класс', u'член_класса', u'antonym'):
                        continue

                    if word1 == u'быть' or word2 == u'быть':
                        continue

                    if word1 != word2 and word1:  # in all_words and word2 in all_words:
                        if word1 not in self.word2links:
                            self.word2links[word1] = []
                        self.word2links[word1].append((word2, pos2, relat))

        self.word2links[u'ты'] = [(u'твой', u'ПРИЛАГАТЕЛЬНОЕ', u'в_прил')]
        self.word2links[u'я'] = [(u'мой', u'ПРИЛАГАТЕЛЬНОЕ', u'в_прил')]
        self.word2links[u'мы'] = [(u'наш', u'ПРИЛАГАТЕЛЬНОЕ', u'в_прил')]
        self.word2links[u'вы'] = [(u'ваш', u'ПРИЛАГАТЕЛЬНОЕ', u'в_прил')]
        logging.info('{} items in thesaurus loaded'.format(len(self.word2links)))

    def get_linked(self, word1):
        res = []
        if word1 in self.word2links:
            for link in self.word2links[word1]:
                res.append(link)
        return res

    def are_linked(self, lemma1, lemma2):
        if lemma1 in self.word2links:
            if lemma2 in self.word2links:
                for link in self.word2links[lemma2]:
                    if link[0] == lemma1:
                        if link[2] not in (u'в_класс', u'член_класса'):
                            return True

        return False


class GrammarDict:
    def __init__(self):
        pass

    def split_tag(self, tag):
        return tuple(tag.split(':'))

    def split_tags(self, tags_str):
        return [self.split_tag(tag) for tag in tags_str.split(' ')]

    def is_good(self, tags_str):
        # Исключаем краткие формы прилагательных в среднем роде, так как
        # они обычно омонимичны с более употребимыми наречиями.
        return u'КРАТКИЙ:1 ПАДЕЖ:ИМ РОД:СР' not in tags_str

    def load(self, path, allowed_words):
        self.word2pos = dict()
        self.word2tags = dict()
        self.word_pos2tags = dict()
        self.tagstr2id = dict()
        self.tagsid2list = dict()
        logging.info(u'Loading morphology information from {}'.format(path))

        self.word2pos = dict()

        # Второй проход - сохраняем информацию для всех слов, кроме вошедших
        # в список неоднозначных.
        with codecs.open(path, 'r', 'utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 5:
                    score = 0
                    if tx[4] != 'NULL':
                        score = int(tx[4])
                        if score < 0:
                            # пропускаем формы, которые помечены как редкие или неграмматические (частотность < 0)
                            #print(u'{}'.format(tx[0]))
                            continue

                    word = tx[0].replace(u' - ', u'-').lower()

                    # начало отладки
                    #if word == u'кушаешь':
                    #    print('DEBUG@481')
                    # конец отладки

                    tags_str = tx[3]
                    if self.is_good(tags_str):
                        if allowed_words is not None and word not in allowed_words:
                            continue

                        tags_str = tags_str.replace(u'ПЕРЕЧИСЛИМОСТЬ:ДА', u'')\
                            .replace(u'ПЕРЕЧИСЛИМОСТЬ:НЕТ', u'')\
                            .replace(u'ПЕРЕХОДНОСТЬ:ПЕРЕХОДНЫЙ', u'')\
                            .replace(u'ПЕРЕХОДНОСТЬ:НЕПЕРЕХОДНЫЙ', u'')

                        if True:  #word in all_words:
                            pos0 = tx[1]
                            pos = decode_pos(pos0)
                            if pos0 == u'ИНФИНИТИВ':
                                tags_str += u' ФОРМА_ГЛАГОЛА:ИНФИНИТИВ'
                            elif pos0 == u'ГЛАГОЛ':
                                tags_str += u' ФОРМА_ГЛАГОЛА:ГЛАГОЛ'
                            elif pos0 == u'ДЕЕПРИЧАСТИЕ':
                                tags_str += u' ФОРМА_ГЛАГОЛА:ДЕЕПРИЧАСТИЕ'

                            self.add_word(word, pos, tags_str)

            # отдельно добавляем фиктивную информацию для местоимений в 3м лице, чтобы
            # они могли меняться на существительные
            s_noun = u'СУЩЕСТВИТЕЛЬНОЕ'
            self.add_word(u'никто', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:МУЖ')
            self.add_word(u'ничто', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:МУЖ')
            self.add_word(u'он', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:МУЖ')
            self.add_word(u'она', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:ЖЕН')
            self.add_word(u'оно', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:ЕД РОД:СР')
            self.add_word(u'они', s_noun, u'ПАДЕЖ:ИМ ЧИСЛО:МН')
            self.add_word(u'тебе', s_noun, u'ПАДЕЖ:ДАТ ЧИСЛО:ЕД')
            self.add_word(u'тебя', s_noun, u'ПАДЕЖ:ВИН ЧИСЛО:ЕД')
            self.add_word(u'тобой', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'тобою', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'мне', s_noun, u'ПАДЕЖ:ДАТ ЧИСЛО:ЕД')
            self.add_word(u'меня', s_noun, u'ПАДЕЖ:ВИН ЧИСЛО:ЕД')
            self.add_word(u'мной', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'мною', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'нам', s_noun, u'ПАДЕЖ:ДАТ ЧИСЛО:ЕД')
            self.add_word(u'нами', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'вам', s_noun, u'ПАДЕЖ:ДАТ ЧИСЛО:ЕД')
            self.add_word(u'вами', s_noun, u'ПАДЕЖ:ТВОР ЧИСЛО:ЕД')
            self.add_word(u'вас', s_noun, u'ПАДЕЖ:ВИН ЧИСЛО:ЕД')
            logging.info('Total number of wordforms={}/{}'.format(len(self.word2pos), len(self.word2tags)))

    def add_word(self, word, pos, tags_str0):
        self.word2pos[word] = pos
        #if pos == u'СУЩЕСТВИТЕЛЬНОЕ':
        #    self.nouns.add(word)
        #elif pos == u'ПРИЛАГАТЕЛЬНОЕ':
        #    self.adjs.add(word)
        #elif pos == u'ГЛАГОЛ':
        #    self.verbs.add(word)
        #elif pos == u'НАРЕЧИЕ':
        #    self.adverbs.add(word)

        tags_str = u'ЧАСТЬ_РЕЧИ:'+pos + u' ' + tags_str0

        # формы прилагательных в винительном падеже дополняем тегами ОДУШ:ОДУШ и ОДУШ:НЕОДУШ, если не указан тег ОДУШ:НЕОДУШ
        if pos == u'ПРИЛАГАТЕЛЬНОЕ' and u'ПАДЕЖ:ВИН' in tags_str and u'ОДУШ:' not in tags_str:
            tags_str += u' ОДУШ:ОДУШ ОДУШ:НЕОДУШ'

        tags_str = tags_str.replace(u'  ', u' ')
        if tags_str not in self.tagstr2id:
            tags_id = len(self.tagstr2id)
            self.tagstr2id[tags_str] = tags_id
            self.tagsid2list[tags_id] = self.split_tags(tags_str)
        else:
            tags_id = self.tagstr2id[tags_str]

        if word not in self.word2tags:
            self.word2tags[word] = [tags_id]
        else:
            self.word2tags[word].append(tags_id)

        word_pos = (word, pos)
        if word_pos not in self.word_pos2tags:
            self.word_pos2tags[word_pos] = [tags_id]
        else:
            self.word_pos2tags[word_pos].append(tags_id)

    def __contains__(self, word):
        return word in self.word2pos

    def get_pos(self, word):
        if word in self.word2pos:
            return self.word2pos[word]
        else:
            return None

    def get_word_tagsets(self, word):
        tagsets = []
        for tagset_id in self.word2tags[word]:
            tagsets.append(self.tagsid2list[tagset_id])
        return tagsets

    def get_word_tagsets2(self, word, part_of_speech):
        tagsets = []
        k = (word, part_of_speech)
        if k in self.word_pos2tags:
            for tagset_id in self.word_pos2tags[k]:
                tagsets.append(self.tagsid2list[tagset_id])
        return tagsets


class GT_Item(object):
    def __init__(self):
        pass

    def generate(self, topic_words, gren):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()


class GT_Word(GT_Item):
    def __init__(self, word):
        if not all((c.lower() in u'1234567890абвгдеёжзийклмнопрстуфхцчшщъыьэюя?.!-:;_') for c in word):
            msg = u'Invalid word '+word
            print(msg)
            raise ValueError(msg)

        super(GT_Word, self).__init__()
        self.word = word

    def generate(self, topic_words, gren):
        return [(self.word, 1.0)]

    def __repr__(self):
        return self.word

    def __eq__(self, other):
        return other is GT_Word and self.word == other.word


class GT_RandomWord(GT_Item):
    def __init__(self, words_str):
        super(GT_RandomWord, self).__init__()
        words = words_str[1:-1].split('|')
        assert(len(words) >= 2)
        self.words = words

    def generate(self, topic_words, gren):
        return [(random.choice(self.words), 1.0)]

    def __repr__(self):
        return u'{' + u' '.join(self.words) + u'}'

    def __eq__(self, other):
        return other is GT_RandomWord and self.words == other.words


class GT_RegexWordFilter(GT_Item):
    def __init__(self, rx_mask):
        super(GT_RegexWordFilter, self).__init__()
        self.rx_mask = rx_mask[1:-1]
        self.rx = re.compile(self.rx_mask, re.IGNORECASE)

    def __repr__(self):
        return u'('+self.rx_mask+u')'

    def __eq__(self, other):
        return other is GT_RegexWordFilter and self.rx_mask == other.rx_mask

    def generate(self, topic_words, gren):
        selected_forms = []

        for topic_word in topic_words:
            for variant in topic_word.get_all_variants():
                if self.rx.match(variant.word):
                    selected_forms.append((variant.word, variant.weight))
                    break

        return selected_forms[:5]



class GT_NamedSet(GT_Item):
    def __init__(self, set_name, words):
        super(GT_NamedSet, self).__init__()
        self.set_name = set_name
        self.words = words

    def __repr__(self):
        return self.set_name

    def __eq__(self, other):
        return other is GT_NamedSet and self.set_name == other.set_name

    def generate(self, topic_words, gren):
        selected_forms = []

        for word in self.words:
            selected_forms.append((word, 1.0))
            break

        return selected_forms



class GT_Replaceable(GT_Item):
    def __init__(self, tags):
        super(GT_Replaceable, self).__init__()
        self.src_tags = u'['+tags+u']'
        self.part_of_speech = None
        self.tags = []
        for tag in tags.replace(',', ' ').split(' '):
            if tag == u'сущ':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'СУЩЕСТВИТЕЛЬНОЕ'))
            elif tag == u'союз':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'СОЮЗ'))
            elif tag == u'местоим':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'МЕСТОИМЕНИЕ'))
            elif tag == u'числит':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'ЧИСЛИТЕЛЬНОЕ'))
            elif tag == u'гл':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'ГЛАГОЛ'))
            elif tag == u'инф':
                self.tags.append((u'ФОРМА_ГЛАГОЛА', u'ИНФИНИТИВ'))
            elif tag == u'деепр':
                self.tags.append((u'ФОРМА_ГЛАГОЛА', u'ДЕЕПРИЧАСТИЕ'))
            elif tag == u'прил':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'ПРИЛАГАТЕЛЬНОЕ'))
            elif tag == u'нареч':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'НАРЕЧИЕ'))
            elif tag == u'предлог':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'ПРЕДЛОГ'))
            elif tag == u'num_word':
                self.tags.append((u'ЧАСТЬ_РЕЧИ', u'NUM_WORD'))
            elif tag == u'ед':
                self.tags.append((u'ЧИСЛО', u'ЕД'))
            elif tag == u'мн':
                self.tags.append((u'ЧИСЛО', u'МН'))
            elif tag == u'им':
                self.tags.append((u'ПАДЕЖ', u'ИМ'))
            elif tag == u'вин':
                self.tags.append((u'ПАДЕЖ', u'ВИН'))
            elif tag == u'род':
                self.tags.append((u'ПАДЕЖ', u'РОД'))
            elif tag == u'тв':
                self.tags.append((u'ПАДЕЖ', u'ТВОР'))
            elif tag == u'дат':
                self.tags.append((u'ПАДЕЖ', u'ДАТ'))
            elif tag == u'предл':
                self.tags.append((u'ПАДЕЖ', u'ПРЕДЛ'))
            elif tag == u'муж':
                self.tags.append((u'РОД', u'МУЖ'))
            elif tag == u'жен':
                self.tags.append((u'РОД', u'ЖЕН'))
            elif tag == u'ср':
                self.tags.append((u'РОД', u'СР'))
            elif tag == u'наст':
                self.tags.append((u'ВРЕМЯ', u'НАСТОЯЩЕЕ'))
            elif tag == u'прош':
                self.tags.append((u'ВРЕМЯ', u'ПРОШЕДШЕЕ'))
            elif tag == u'2':
                self.tags.append((u'ЛИЦО', u'2'))
            elif tag == u'3':
                self.tags.append((u'ЛИЦО', u'3'))
            elif tag == u'~кр':
                self.tags.append((u'КРАТКИЙ', u'0'))
            elif tag == u'кр':
                self.tags.append((u'КРАТКИЙ', u'1'))
            elif tag == u'положит':
                self.tags.append((u'СТЕПЕНЬ', u'АТРИБ'))
            elif tag == u'сравн':
                self.tags.append((u'СТЕПЕНЬ', u'СРАВН'))
            elif tag == u'превосх':
                self.tags.append((u'СТЕПЕНЬ', u'ПРЕВОСХ'))
            elif tag == u'одуш':
                self.tags.append((u'ОДУШ', u'ОДУШ'))
            elif tag == u'неодуш':
                self.tags.append((u'ОДУШ', u'НЕОДУШ'))
            elif tag == u'сов':
                self.tags.append((u'ВИД', u'СОВЕРШ'))
            elif tag == u'несов':
                self.tags.append((u'ВИД', u'НЕСОВЕРШ'))
            elif tag == u'изъяв':
                self.tags.append((u'НАКЛОНЕНИЕ', u'ИЗЪЯВ'))
            elif tag == u'побуд':
                self.tags.append((u'НАКЛОНЕНИЕ', u'ПОБУД'))
            elif tag == u'модиф_прил':
                self.tags.append((u'ТИП_МОДИФ', u'ПРИЛ'))
            elif tag == u'модиф_глаг':
                self.tags.append((u'ТИП_МОДИФ', u'ГЛАГ'))
            elif tag == u'модиф_нареч':
                self.tags.append((u'ТИП_МОДИФ', u'НАРЕЧ'))
            elif tag == u'модиф_сущ':
                self.tags.append((u'ТИП_МОДИФ', u'СУЩ'))
            elif tag == u'модиф_предл':
                self.tags.append((u'ТИП_МОДИФ', u'ПРЕДЛ'))
            else:
                err = u'Unknown tag "{}" in tag string "{}"'.format(tag, tags)
                print(u'ERROR: {}'.format(err))
                raise NotImplementedError(err)

        for tag, tag_val in self.tags:
            if tag == u'ЧАСТЬ_РЕЧИ':
                self.part_of_speech = tag_val
                break

    def __repr__(self):
        return self.src_tags

    def __eq__(self, other):
        return other is GT_Replaceable and self.src_tags == other.src_tags

    def generate(self, topic_words, gren):
        selected_forms = []

        for topic_word in topic_words:
            for variant in topic_word.get_variants(self.part_of_speech):
                #if variant.word in gren:
                for tagset in variant.tagsets:
                    #if (u'ЧАСТЬ_РЕЧИ', variant.part_of_speech) in tagset:
                    if all((tag in tagset) for tag in self.tags):
                        selected_forms.append((variant.word, variant.weight))
                        break

        best_forms = sorted(selected_forms, key=lambda z: -z[1])[:5]
        return best_forms


class GeneratedPhrase(object):
    def __init__(self, words, proba):
        self.words = words
        self.words_hash = tuple(words).__hash__()
        self.proba0 = proba
        self.total_proba = proba

    def get_str(self):
        return clean_output(u' '.join(self.words))

    def get_proba0(self):
        return self.proba0

    def get_rank(self):
        return self.total_proba

    def set_rank(self, p):
        self.total_proba = p

    def __repr__(self):
        return u'{} ({})'.format(self.get_str(), self.get_rank())

    def __hash__(self):
        return self.words_hash

    def __eq__(self, y):
        # Могут быть одинаковые фразы с разными вероятностями, их мы
        # рассматриваем как одинаковые.
        return self.words == y.words


class GF_Node(object):
    def __init__(self, gt_item):
        #assert(gt_item is GT_Item)
        self.item = gt_item
        self.is_terminal = False
        self.next_items = []

    def set_terminal(self):
       self.is_terminal = True

    def find_next(self, gt_item):
        for next_item in self.next_items:
            if next_item.item == gt_item:
                return next_item

        return None

    def merge(self, gt_items):
        n_gt_items = len(gt_items)
        head_item = gt_items[0]
        mapped_next = self.find_next(head_item)
        if mapped_next:
            if n_gt_items == 1:
                mapped_next.set_terminal()
            else:
                mapped_next.merge(gt_items[1:])
        else:
            new_item = GF_Node(head_item)
            self.next_items.append(new_item)
            if n_gt_items > 1:
                new_item.merge(gt_items[1:])
            else:
                new_item.set_terminal()

    def stat(self):
        child_nodes = 1
        terminal_nodes = self.is_terminal
        for item in self.next_items:
            a, b = item.stat()
            child_nodes += a
            terminal_nodes += b

        return child_nodes, terminal_nodes

    def __repr__(self):
        if self.item:
            return u'GF_Node('+self.item.__repr__()+u')'
        else:
            return u'root'

    def generate(self, prev_forms_slots, beam_searcher, topic_words, gren, ngrams):
        if self.item:
            forms_in_slot = self.item.generate(topic_words, gren)
            if len(forms_in_slot) == 0:
                # Пустой слот означает, что нет ни одного варианта генерации для элемента,
                # и далее по узлам идти нет смысла
                return
            else:
                new_forms_slots = prev_forms_slots + [forms_in_slot]
                if self.is_terminal:
                    # Терминальный узел - собран контекст для генерации цепочки слов фразы.
                    beam_searcher(new_forms_slots)

                # Продолжаем генерацию на последующих (справа) узлах
                for next_node in self.next_items:
                    next_node.generate(new_forms_slots, beam_searcher, topic_words, gren, ngrams)
        else:
            for next_node in self.next_items:
                next_node.generate(prev_forms_slots, beam_searcher, topic_words, gren, ngrams)


class GF_RootNode(GF_Node):
    def __init__(self):
        super(GF_RootNode, self).__init__(None)


class TopicWordVariant(object):
    def __init__(self, word, part_of_speech, tagsets, weight):
        self.word = word
        self.part_of_speech = part_of_speech
        self.tagsets = tagsets
        self.weight = weight


class TopicWord(object):
    def __init__(self, word, proba):
        self.word = word
        self.slot_proba = proba
        self.pos2variants = dict()
        self.variant_words = set()

    def add_variant(self, variant):
        if variant.part_of_speech not in self.pos2variants:
            self.pos2variants[variant.part_of_speech] = []

        self.pos2variants[variant.part_of_speech].append(variant)
        self.variant_words.add(variant.word)

    def get_variants(self, part_of_speech):
        return self.pos2variants.get(part_of_speech, [])

    def get_all_variants(self):
        for pos in self.pos2variants.keys():
            for v in self.pos2variants[pos]:
                yield v

    def find_in_variants(self, generated_phrase_words_set):
        return len(self.variant_words & generated_phrase_words_set)

    def __repr__(self):
        return u'{} ({}) => {} variant(s)'.format(self.word, self.slot_proba, len(self.variant_words))


def is_numword(s):
    return all(c in '0123456789' for c in s)


def construct_topic_word(word, word_class, slot_proba, corpus, thesaurus, lexicon, grdict, assocs):
    topic_word = TopicWord(word, slot_proba)

    if is_numword(word):
        topic_word.add_variant(TopicWordVariant(word, u'NUM_WORD', [[(u'ЧАСТЬ_РЕЧИ', u'NUM_WORD')]], slot_proba))
    else:
        lemmas = dict()

        for lemma0, pos0 in lexicon.get_lemmas(word):
            if word_class is not None and pos0 != word_class:
                continue

            lemmas[(lemma0, pos0)] = 1.0

            for lemma2, pos2, link_type in thesaurus.get_linked(lemma0):
                if link_type not in (u'в_класс', u'член_класса'):
                    k = (lemma2, pos2)
                    lemmas[k] = max(0.8, lemmas.get(k, 0.0))

                    # Добавляем синонимы синонимов и т.д.
                    for lemma3, pos3, link_type3 in thesaurus.get_linked(lemma2):
                        if pos3 is None:
                            raise RuntimeError()
                        k = (lemma3, pos3)
                        lemmas[k] = max(0.6, lemmas.get(k, 0.0))

        all_lemmas = dict()
        for (lemma, pos), relat in lemmas.items():
            # НАЧАЛО ОТЛАДКИ
            #if lemma == u'минута':
            #    print('DEBUG@995')
            # КОНЕЦ ОТЛАДКИ
            if pos is None:
                all_lemmas[(lemma, u'<unknown>')] = relat
            else:
                all_lemmas[(lemma, pos)] = relat

            if False:
                ax = assocs.get_assocs(lemma)
                if len(ax) > 0:
                    max_relat = max(map(operator.itemgetter(1), ax))
                    for lemma2, relat2 in ax:
                        pos2 = grdict.get_pos(lemma2)
                        if pos2:
                            k2 = (lemma2, pos2)
                            w = 0.9*relat2/max_relat
                            all_lemmas[k2] = max(relat*w, all_lemmas.get(k2, 0.0))

        all_forms = dict()
        for (lemma, part_of_speech), relat in all_lemmas.items():
            # НАЧАЛО ОТЛАДКИ
            #if lemma == u'минута':
            #    print('DEBUG@1014')
            # КОНЕЦ ОТЛАДКИ

            lforms = lexicon.get_forms(lemma, part_of_speech)
            for form in lforms:
                if form in corpus:
                    form_key = (form, part_of_speech)
                    all_forms[form_key] = max(relat, all_forms.get(form_key, -1))

        for (word, part_of_speech), p in sorted(list(all_forms.items()), key=lambda z: -z[1]):
            tagsets = grdict.get_word_tagsets2(word, part_of_speech)
            topic_word.add_variant(TopicWordVariant(word, part_of_speech, tagsets, p*slot_proba))

    return topic_word


def merge_lists(lists):
    return itertools.chain(*lists)


class Macro(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right.split(u' ')


class Macros(object):
    def __init__(self):
        self.name2macros = dict()

    def explode_right(self, right):
        tokens = right.split(' ')
        token_slots = []
        for token in tokens:
            slot = []
            if token in self.name2macros:
                for macro in self.name2macros[token]:
                    slot.append(macro.right)
            else:
                slot.append([token])
            token_slots.append(slot)

        comb = itertools.product(*token_slots)
        exploded_rights = [u' '.join(merge_lists(tokens)) for tokens in comb]
        return exploded_rights


    def parse(self, line):
        parts = line.split('=')
        left = parts[0].strip()
        right = parts[1].strip()
        rights = self.explode_right(right)

        for right in rights:
            macro = Macro(left, right)
            if left in self.name2macros:
                self.name2macros[left].append(macro)
            else:
                self.name2macros[left] = [macro]

    def __contains__(self, token):
        return token in self.name2macros

    def __getitem__(self, token):
        return self.name2macros[token]


class GenerativeTemplate(object):
    def __init__(self):
        self.items = []

    def __repr__(self):
        return u' '.join(unicode(item) for item in self.items)

    def generate(self, topic_words, gren, thesaurus, lexicon, ngrams):
        forms_in_slots = [item.generate(topic_words, gren) for item in self.items]
        if any((len(f) == 0) for f in forms_in_slots):
            # Пустой слот означает, что нет ни одного варианта генерации для элемента
            return []

        # начало отладки
        #for word_slot in forms_in_slots:
        #    for r in word_slot:
        #        if not isinstance(r, (list, tuple)) or len(r) != 2:
        #            print(u'ERROR@87 data={}'.format(r))
        #            exit(0)
        # конец отладки

        paths = beam_search(forms_in_slots, thesaurus, lexicon, ngrams)
        return [GeneratedPhrase(path, proba) for (path, proba) in paths]

    def __len__(self):
        return len(self.items)


class GenerativeTemplates(object):
    def __init__(self):
        self.templates = []
        self.flow_root = None

    def parse(self, s, macros, named_sets, max_rule_len):
        terms = s.split(u' ')
        if len(terms) > max_rule_len:
            return

        # Каждый терм в исходной строке описания шаблона может быть раскрыт в несколько вариантов цепочек токенов
        term_slots = []
        for term in terms:
            slot = []
            if term in macros:
                for macro in macros[term]:
                    slot.append(macro.right)
            else:
                slot.append([term])

            term_slots.append(slot)

        templates = []

        # Теперь генерируем все сочетания элементов в слотах
        # и получаем множество шаблонов
        for variant in itertools.product(*term_slots):
            template = GenerativeTemplate()
            for term_sequence in variant:
                for term in term_sequence:
                    if len(term) > 3 and term[0] == u'[' and term[-1] == u']':
                        template.items.append(GT_Replaceable(term[1:-1]))
                    elif len(term) >= 3 and term[0] == u'{' and term[-1] == u'}':
                        template.items.append(GT_RandomWord(term))
                    elif len(term) >= 3 and term[0] == u'(' and term[-1] == u')':
                        template.items.append(GT_RegexWordFilter(term))
                    elif term in named_sets:
                        template.items.append(GT_NamedSet(term, named_sets[term]))
                    else:
                        template.items.append(GT_Word(term))

            if len(template.items) <= max_rule_len:
                templates.append(template)

        self.templates.extend(templates)

    def compile(self, max_len):
        self.flow_root = GF_RootNode()
        for template in self.templates:
            if len(template) <= max_len:
                self.flow_root.merge(template.items)

        nb_nodes, nb_terminals = self.flow_root.stat()
        print('nb_nodes={} nb_terminals={}'.format(nb_nodes, nb_terminals))


    @staticmethod
    def beam_searcher(forms_in_slots, thesaurus, lexicon, ngrams, generated_phrases):
        paths = beam_search(forms_in_slots, thesaurus, lexicon, ngrams)

        # Одна и та же фраза может быть сгенерирована несколько раз с разными
        # весами. Мы оставим вариант с самым большим весом.
        for path, proba in paths:
            phrase = GeneratedPhrase(path, proba)
            generated_phrases[phrase] = max(generated_phrases.get(phrase, -1e8), proba)


    def generate_phrases(self, topic_words, grdict, thesaurus, lexicon, all_ngrams):
        all_generated_phrases = dict()  # сгенерированные фразы и их веса

        if self.flow_root:
            # Используем generation graph для ускорения
            prev_forms_slots = []
            self.flow_root.generate(prev_forms_slots,
                                    lambda forms_in_slots: GenerativeTemplates.beam_searcher(forms_in_slots, thesaurus, lexicon, all_ngrams, all_generated_phrases),
                                    topic_words, grdict, all_ngrams)
        else:
            # Перебор всех шаблонов
            for template in self.templates:
                generated_phrases = template.generate(topic_words, grdict, all_ngrams)
                all_generated_phrases.update(generated_phrases)

        return list(all_generated_phrases.keys())


class BeamSearchItem(object):
    def __init__(self, word, word_proba, path_proba, prev_item):
        self.word = word
        self.word_proba = word_proba
        self.path_proba = path_proba
        self.prev_item = prev_item

        if self.prev_item:
            self.path_words = self.prev_item.path_words + [self.word]
        else:
            self.path_words = [self.word]

    def get_path_words(self):
        return self.path_words

    def get_path_proba(self):
        return self.path_proba


def normalize_proba(x):
    return 1 / (1 + math.exp(-x))


def calc_discount(new_word, path_words, thesaurus, lexicon):
    # начало отладки
    #if new_word == u'кошку' and u'ловит' in path_words and u'киса' in path_words:
    #    pass
    # конец отладки

    if new_word in path_words:
        return 0.1

    new_lemma = lexicon.get_lemma(new_word)

    for path_word in path_words:
        path_lemma = lexicon.get_lemma(path_word)
        if path_lemma == new_lemma:
            return 0.2

        if thesaurus.are_linked(new_lemma, path_lemma):
            return 0.3

    return 1.0


def same_stem(word1, word2):
    return word1[0:3] == word2[0:3]


def beam_search(word_slots, thesaurus, lexicon, ngrams):
    beam_size = 4
    cur_items = []

    # Для затравки - берем несколько лучших слов из первого слота
    for word, word_proba in word_slots[0][:beam_size*5]:
        cur_item = BeamSearchItem(word, word_proba, word_proba, None)
        cur_items.append(cur_item)

    for word_slot in word_slots[1:]:
        next_items = []

        # начало отладки
        #for r in word_slot:
        #    if not isinstance(r, (list, tuple)) or len(r) != 2:
        #        print(u'ERROR@981 data={}'.format(r))
        #        exit(0)
        # конец отладки

        for word, word_proba in word_slot:
            for cur_item in cur_items:
                if cur_item.word == word:
                    # буквальный повтор слова запрещаем безусловно
                    continue

                # НАЧАЛО ОТЛАДКИ
                #if cur_item.word == u'как' and word == u'ты':
                #    print('DEBUG@1260')
                # КОНЕЦ ОТЛАДКИ

                # За повтор леммы - штрафуем.
                trans_proba1 = 0.5 if same_stem(cur_item.word, word) and lexicon.same_lemma(cur_item.word, word) else 1.0

                # встречается ли такая 2-грамма
                transition_proba = 1.0 if (cur_item.word, word) in ngrams else 0.5

                discount = calc_discount(word, cur_item.path_words, thesaurus, lexicon)

                path_proba = discount * trans_proba1 * transition_proba * word_proba * cur_item.path_proba
                item = BeamSearchItem(word, word_proba, path_proba, cur_item)
                next_items.append(item)
        cur_items = sorted(next_items, key=lambda z: -z.path_proba)[:beam_size]

    paths = [(item.get_path_words(), item.get_path_proba()) for item in cur_items]
    return paths


def calc_phrase_score(phrase, topic_words, ngrams, assocs, max_gap):
    words = phrase.words
    p = 1.0

    p += len(words) * 1e-3  # небольшая награда за более длинные предложения

    # Штрафуем фразы, в которых не использованы какие-то варианты для исходных слов.
    words_set = set(words)
    for topic_slot in topic_words:
        if not topic_slot.find_in_variants(words_set):
            p -= 0.3 * topic_slot.slot_proba

    for gap in range(max_gap):
        if ngrams.has_2grams:
            for word1, word2 in zip(words, words[1:]):
                # Штраф за неизвестное сочетание двух слов
                if (word1, word2) not in ngrams:
                    p -= 1.0

                # Поощрение за известное сочетание двух слов
                if (word1, word2) in ngrams:
                    p += 1.0

        if ngrams.has_3grams:
            for word1, word2, word3 in zip(words, words[1:], words[2:]):
                # Штраф за неизвестное сочетание трех слов
                if (word1, word2, word3) not in ngrams:
                    p -= 0.5

                # Поощрение за известное сочетание трех слов
                if (word1, word2, word3) in ngrams:
                    p += 2.0

    #for word1, word2 in zip(words, words[1:]):
        #score1 += ngrams.get((word1, word2))
        #score2 += assocs.get_mi(word1, word2)

    #return score1 * score2
    return p




class GenerativeGrammarDictionaries(object):
    def __init__(self):
        self.all_ngrams = None
        self.assocs = None
        self.thesaurus = None
        self.lexicon = None
        self.grdict = None

    def prepare(self, data_folder, max_ngram_gap, use_thesaurus=True, use_assocs=True,
                corpora_paths=None, lexicon_words=None):
        self.max_ngram_gap = max_ngram_gap

        if corpora_paths is None:
            corpora = [os.path.join(data_folder, 'pqa_all.dat'),
                       os.path.join(data_folder, 'ngrams_corpus.txt'),
                       os.path.join(data_folder, 'paraphrases.txt'),
                       # r'/media/inkoziev/corpora/Corpus/word2vector/ru/w2v.ru.corpus.txt'
                       ]
        else:
            corpora = corpora_paths

        self.all_ngrams = NGrams()
        self.all_ngrams.collect(corpora, max_gap=max_ngram_gap, max_2grams=5000000, max_3grams=5000000)

        self.assocs = Associations()
        # self.assocs.load(os.path.join(data_folder, 'dict/mutual_info_2_ru.dat'), grdict)
        if use_assocs:
            self.assocs.collect_from_corpus(corpora, 5000000)

        # Большой текстовый корпус, в котором текст уже токенизирован и нормализован.
        #corpus = CorpusWords()
        #corpus.analyze(corpora, os.path.join(data_folder, 'dict/links.csv'),
        #               os.path.join(data_folder, 'word2tags.dat'),
        #               os.path.join(data_folder, 'dict/word2freq_wiki.dat'))
        #corpus.save()

        self.thesaurus = Thesaurus()
        if use_thesaurus:
            self.thesaurus.load(os.path.join(data_folder, 'dict/links.csv'))  # , corpus)

        self.lexicon = Word2Lemmas()
        self.lexicon.load(os.path.join(data_folder, 'dict/word2lemma.dat'), lexicon_words)

        self.grdict = GrammarDict()
        self.grdict.load(os.path.join(data_folder, 'word2tags.dat'), lexicon_words)

    def save(self, filepath):
        logging.info(u'Storing generative grammar dictionaries to "{}"'.format(filepath))
        with open(filepath, 'wb') as f:
            self.pickle_to(f)

    def pickle_to(self, file):
        data = (self.thesaurus, self.lexicon, self.grdict, self.assocs, self.all_ngrams, self.max_ngram_gap)
        pickle.dump(data, file)

    def load(self, filepath):
        logging.info(u'Loading generative grammar dictionaries from "{}"'.format(filepath))
        with open(filepath, 'rb') as f:
            self.thesaurus, self.lexicon, self.grdict, self.assocs, self.all_ngrams, self.max_ngram_gap = pickle.load(f)

    def get_random_verb(self, known_words):
        return self.lexicon.get_random_verb(known_words)

    def get_random_noun(self, known_words):
        return self.lexicon.get_random_noun(known_words)

    def get_random_adj(self, known_words):
        return self.lexicon.get_random_adj(known_words)

    def get_random_adv(self, known_words):
        return self.lexicon.get_random_adv(known_words)


class GenerativeGrammarEngine(object):
    def __init__(self):
        self.named_sets = dict()
        self.macros = Macros()
        self.templates = GenerativeTemplates()
        self.max_rule_len = 8
        self.wordbag_words = set()
        self.dictionaries = None

    def set_max_rule_length(self, max_len):
        self.max_rule_len = max_len

    def set_dictionaries(self, dictionaries):
        self.dictionaries = dictionaries

    def save(self, filepath):
        logging.info(u'Storing grammar to "{}"'.format(filepath))
        with open(filepath, 'wb') as f:
            self.pickle_to(f)

    def pickle_to(self, file):
        pickle.dump(self.templates, file)
        pickle.dump(self.wordbag_words, file)
        pickle.dump(self.named_sets, file)

    @staticmethod
    def unpickle_from(file):
        grammar = GenerativeGrammarEngine()
        grammar.templates = pickle.load(file)
        grammar.wordbag_words = pickle.load(file)
        grammar.named_sets = pickle.load(file)
        return grammar

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.templates = pickle.load(f)

    def add_named_set(self, set_name, words):
        if set_name not in self.named_sets:
            self.named_sets[set_name] = set(words)
        else:
            self.named_sets[set_name].update(words)

    def add_word(self, word, part_of_speech):
        self.wordbag_words.add((word, part_of_speech))

    def add_macro(self, macro_str):
        self.macros.parse(macro_str)

    def add_rule(self, rule_str):
        self.templates.parse(rule_str, self.macros, self.named_sets, self.max_rule_len)

    def compile_rules(self):
        self.templates.compile(self.max_rule_len)

    def generate(self, words_bag, known_words):
        return self.generate2([(word, 1.0) for word in words_bag], known_words)

    def generate2(self, words_p_bag, known_words):
        topic_words = list()
        for word, proba in words_p_bag:
            topic_word = construct_topic_word(word, None, proba, known_words,
                                              self.dictionaries.thesaurus,
                                              self.dictionaries.lexicon,
                                              self.dictionaries.grdict,
                                              self.dictionaries.assocs)
            topic_words.append(topic_word)

        for word, part_of_speech in self.wordbag_words:
            topic_word = construct_topic_word(word, part_of_speech, 1.0, known_words,
                                              self.dictionaries.thesaurus,
                                              self.dictionaries.lexicon,
                                              self.dictionaries.grdict,
                                              self.dictionaries.assocs)
            topic_words.append(topic_word)

        all_generated_phrases = self.templates.generate_phrases(topic_words,
                                                                self.dictionaries.grdict,
                                                                self.dictionaries.thesaurus,
                                                                self.dictionaries.lexicon,
                                                                self.dictionaries.all_ngrams)

        weighted_phrases = []
        for phrase in all_generated_phrases:
            # НАЧАЛО ОТЛАДКИ
            #if phrase.get_str() in (u'Илья', u'Папа'):
            #    print('DEBUG@1331')
            # КОНЕЦ ОТЛАДКИ

            p2 = calc_phrase_score(phrase, topic_words, self.dictionaries.all_ngrams,
                                   self.dictionaries.assocs,
                                   self.dictionaries.max_ngram_gap)
            p2 = normalize_proba(p2) * phrase.get_proba0()
            phrase.set_rank(p2)
            weighted_phrases.append(phrase)

        return weighted_phrases


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    data_folder = '../../../data'
    tmp_folder = '../../../tmp'

    dictionaries = GenerativeGrammarDictionaries()
    dictionaries.prepare(data_folder, tmp_folder)
    dictionaries.save(os.path.join(tmp_folder, 'generative_grammar_dictionaries.bin'))

# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки модели проверки релевантности
ответа для заданных предпосылок и вопроса

Для вопросно-ответной системы https://github.com/Koziev/chatbot.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import gensim
import io
import math
import numpy as np
import argparse
import logging
import random

import rupostagger
import rulemma
import ruword2tags

from ruchatbot.utils.tokenizer import Tokenizer
import ruchatbot.utils.logging_helpers


def decode_pos(pos):
    if pos in [u'ДЕЕПРИЧАСТИЕ', u'ГЛАГОЛ', u'ИНФИНИТИВ']:
        return u'ГЛАГОЛ'
    else:
        return pos


bad_tokens = set(u'? . !'.split())


def clean_phrase(s, tokenizer):
    return u' '.join(filter(lambda w: w not in bad_tokens, tokenizer.tokenize(s))).lower()


class Thesaurus:
    def __init__(self):
        self.word2links = dict()

    def load(self, thesaurus_path):
        logging.info(u'Loading thesaurus from "{}"'.format(thesaurus_path))
        with io.open(thesaurus_path, 'r', encoding='utf-8') as rdr:
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

                    if word1 != word2 and word1:  # in all_words and word2 in all_words:
                        key = (word1, pos1)
                        if key not in self.word2links:
                            self.word2links[key] = []
                        self.word2links[key].append((word2, pos2, relat))

        logging.info('{} items in thesaurus loaded'.format(len(self.word2links)))

    def get_linked(self, word1, part_of_speech1):
        res = []
        key = (word1, part_of_speech1)
        if key in self.word2links:
            for link in self.word2links[key]:
                res.append(link)
        return res


class Word2Lemmas(object):
    def __init__(self):
        pass

    def load(self, path):
        logging.info(u'Loading lexicon from {}'.format(path))
        self.lemmas = dict()
        self.forms = dict()  # для каждой формы - список лемм с пометками части речи
        self.verbs = set()
        self.nouns = set()
        self.adjs = set()
        self.advs = set()

        with io.open(path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 4:
                    if int(tx[3]) >= 0:
                        form = tx[0].replace(u' - ', u'-').lower()
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
            return [word]

    def is_adj(self, word):
        return word in self.adjs

    def is_noun(self, word):
        return word in self.nouns



class Sample:
    def __init__(self, premises, question, answer, label):
        assert(label in (0, 1))
        assert(len(answer) > 0)
        self.premises = premises[:]
        self.question = question
        self.answer = answer
        self.label = label

    def get_key(self):
        return u'|'.join(itertools.chain(self.premises, [self.question, self.answer]))

    def write(self, wrt):
        for premise in self.premises:
            wrt.write(u'{}\n'.format(premise))
        wrt.write(u'{}\n'.format(self.question))
        wrt.write(u'{}\n'.format(self.answer))
        wrt.write(u'{}\n\n'.format(self.label))


def load_samples(input_paths, tokenizer):
    # Загружаем датасеты, содержащий сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ
    samples = []
    max_nb_premises = 0  # макс. число предпосылок в сэмплах
    max_inputseq_len = 0  # макс. длина предпосылок и вопроса в словах
    all_samples = set()

    all_answers = set()

    for input_path, samples_label in input_paths:
        logging.info('Loading samples from "%s"', input_path)

        with io.open(input_path, 'r', encoding='utf-8') as rdr:
            lines = []
            for line in rdr:

                line = line.strip()
                if len(line) == 0:
                    if len(lines) > 0:
                        if len(lines) in (3, 4):  # пока берем только сэмплы с 1 или 2 предпосылками
                            premises = [l.lower() for l in lines[:-2]]
                            question = lines[-2].lower()
                            answer = lines[-1].lower()

                            if answer == 'да' and random.random() < 0.8:
                                continue

                            max_nb_premises = max(max_nb_premises, len(premises))
                            for phrase in lines:
                                words = tokenizer.tokenize(phrase)
                                max_inputseq_len = max(max_inputseq_len, len(words))

                            key = u'|'.join(lines)
                            if key not in all_samples:
                                all_samples.add(key)
                                sample = Sample(premises, question, answer, samples_label)
                                samples.append(sample)
                                all_answers.add(answer)

                        lines = []
                else:
                    lines.append(clean_phrase(line, tokenizer))

    logging.info('samples.count=%d', len(samples))
    logging.info('max_inputseq_len=%d', max_inputseq_len)
    logging.info('max_nb_premises=%d', max_nb_premises)

    # генерируем негативные сэмплы
    all_answers = list(all_answers)
    neg_samples = []
    for sample in samples:
        if sample.label == 1:
            neg_answer = random.choice(all_answers)
            if neg_answer != sample.answer:
                neg_sample = Sample(sample.premises, sample.question, neg_answer, 0)
                neg_samples.append(neg_sample)

    samples.extend(neg_samples)

    nb_0 = sum((sample.label==0) for sample in samples)
    nb_1 = sum((sample.label==1) for sample in samples)

    logging.info('nb_0={}'.format(nb_0))
    logging.info('nb_1={}'.format(nb_1))

    return samples


def is_adj(word, gren):
    return any(u'ПРИЛАГАТЕЛЬНОЕ' in tagset for tagset in gren[word])


def is_noun(word, gren):
    return any(u'СУЩЕСТВИТЕЛЬНОЕ' in tagset for tagset in gren[word])


def is_prepos(word, gren):
    return any(u'ПРЕДЛОГ' in tagset for tagset in gren[word])


def get_tagval(tags, required_tag):
    for tag in tags:
        if tag.startswith(required_tag):
            return tag.split('=')[1]
    return None


def get_tagvals(tags, required_tag):
    for tag in tags:
        if tag.startswith(required_tag):
            yield tag.split('=')[1]


if __name__ == '__main__':
    data_folder = '../../data'
    tmp_folder = '../../tmp'

    # настраиваем логирование в файл
    ruchatbot.utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'prepare_answer_relevancy_dataset.log'))

    input_paths = [(os.path.join(data_folder, 'nonrelevant_answers_handmade_dataset.txt'), 0),
                   (os.path.join(data_folder, 'pqa_all.dat'), 1), ]

    logging.info('Start "prepare_answer_relevancy_dataset.py"')

    tokenizer = Tokenizer()
    tokenizer.load()

    samples = load_samples(input_paths, tokenizer)

    logging.info('Loading dictionaries...')

    thesaurus = Thesaurus()
    thesaurus.load(os.path.join(data_folder, 'dict/links.csv'))  # , corpus)

    lexicon = Word2Lemmas()
    lexicon.load(os.path.join(data_folder, 'dict/word2lemma.dat'))

    grdict = ruword2tags.RuWord2Tags()
    grdict.load()

    flexer = ruword2tags.RuFlexer()
    flexer.load()

    # Аугментация: генерируем негативных сэмплы через выбор вариантов словоформ, отличающихся
    # от использованных в валидном ответе.
    logging.info('Generating negative samples...')
    all_keys = set(sample.get_key() for sample in samples)
    neg_samples = []
    for sample in samples:
        if sample.label == 1:
            answer_words = tokenizer.tokenize(sample.answer)
            answer_len = len(answer_words)
            if answer_len == 1:
                # Аугментация для однословного ответа.
                # Формы единственного слова, кроме упомянутой в ответе
                for lemma, part_of_speech in lexicon.get_lemmas(answer_words[0]):
                    forms = list(lexicon.get_forms(lemma, part_of_speech))
                    forms = np.random.permutation(forms)[:5]
                    for form in forms:
                        if form != answer_words[0]:
                            sample0 = Sample(sample.premises, sample.question, form, 0)
                            key0 = sample0.get_key()
                            if key0 not in all_keys:
                                all_keys.add(key0)
                                neg_samples.append(sample0)
                    # Смена части речи и любые формы нового слова
                    tmp_samples = []
                    for word2, pos2, relation in thesaurus.get_linked(lemma, part_of_speech):
                        for form in lexicon.get_forms(word2, part_of_speech):
                            if form != answer_words[0]:
                                sample0 = Sample(sample.premises, sample.question, form, 0)
                                key0 = sample0.get_key()
                                if key0 not in all_keys:
                                    all_keys.add(key0)
                                    tmp_samples.append(sample0)

                    neg_samples.extend(np.random.permutation(tmp_samples)[:5])

            elif answer_len == 2:
                if is_prepos(answer_words[0], grdict) and is_noun(answer_words[1], grdict):
                    # Аугментация для словосочетаний ПРЕДЛОГ+СУЩ
                    # Соберем пары ПАДЕЖ+ЧИСЛО, которые уже есть среди вариантов для словосочетания
                    found_cases = []
                    for noun_tagset in grdict[answer_words[1]]:
                        tx = noun_tagset.split()
                        noun_case = get_tagval(tx, u'ПАДЕЖ')
                        found_cases.append(noun_case)

                    # Теперь перебираем все варианды падежа, пропуская уже известные
                    prep_cases = list(get_tagvals(list(grdict[answer_words[0]])[0].split(), u'ПАДЕЖ'))
                    for new_case in u'ИМ РОД ТВОР ВИН ДАТ ПРЕДЛ'.split():
                        if new_case not in found_cases and new_case in prep_cases:
                            for new_noun in flexer.find_forms_by_tags(answer_words[1], [(u'ПАДЕЖ', new_case)]):
                                new_answer = answer_words[0] + u' ' + new_noun
                                if new_answer != sample.answer:
                                    sample0 = Sample(sample.premises, sample.question, new_answer, 0)
                                    key0 = sample0.get_key()
                                    if key0 not in all_keys:
                                        all_keys.add(key0)
                                        neg_samples.append(sample0)


                elif is_adj(answer_words[0], grdict) and is_noun(answer_words[1], grdict):
                    # Аугментация для двусловных словосочетаний ПРИЛ+СУЩ
                    a_tagsets = []
                    for adj_tagset in grdict[answer_words[0]]:
                        tx = adj_tagset.split()
                        a_case = get_tagval(tx, u'ПАДЕЖ')
                        a_num = get_tagval(tx, u'ЧИСЛО')
                        a_tagsets.append((a_case, a_num))

                    # Соберем пары ПАДЕЖ+ЧИСЛО, которые уже есть среди вариантов для словосочетания
                    found_tagsets = []
                    noun_gender = None
                    noun_anim = None
                    for noun_tagset in grdict[answer_words[1]]:
                        tx = noun_tagset.split()
                        noun_case = get_tagval(tx, u'ПАДЕЖ')
                        noun_num = get_tagval(tx, u'ЧИСЛО')
                        noun_gender = get_tagval(tx, u'РОД')
                        noun_anim = get_tagval(tx, u'ОДУШ')
                        pair = (noun_case, noun_num)
                        if pair in a_tagsets:
                            found_tagsets.append(pair)

                    # Теперь перебираем все пары ПАДЕЖ+ЧИСЛО, пропуская уже известные в found_tagsets,
                    # и генерируем соответствующие словоформы
                    for new_case in u'ИМ РОД ТВОР ВИН ДАТ ПРЕДЛ'.split():
                        for new_num in u'ЕД МН'.split():
                            if (new_case, new_num) not in found_tagsets:
                                new_adj_tags = [(u'ЧИСЛО', new_num), (u'ПАДЕЖ', new_case)]
                                if new_num == u'ЕД':
                                    new_adj_tags.append((u'РОД', noun_gender))

                                if new_case == u'ВИН' and noun_gender == u'МУЖ':
                                    new_adj_tags.append((u'ОДУШ', noun_anim))

                                for new_adj in flexer.find_forms_by_tags(answer_words[0], new_adj_tags):
                                    for new_noun in flexer.find_forms_by_tags(answer_words[1], [(u'ЧИСЛО', new_num), (u'ПАДЕЖ', new_case)]):
                                        new_answer = new_adj + u' ' + new_noun
                                        if new_answer != sample.answer:
                                            sample0 = Sample(sample.premises, sample.question, new_answer, 0)
                                            key0 = sample0.get_key()
                                            if key0 not in all_keys:
                                                all_keys.add(key0)
                                                neg_samples.append(sample0)

    neg_samples = np.random.permutation(neg_samples)[:len(samples)*2]

    samples.extend(neg_samples)

    logging.info('Generation completed')

    nb_0 = sum((sample.label == 0) for sample in samples)
    nb_1 = sum((sample.label == 1) for sample in samples)

    logging.info('nb_0=%d', nb_0)
    logging.info('nb_1=%d', nb_1)

    # Сохраняем общий получившийся датасет
    with io.open(os.path.join(data_folder, 'answer_relevancy_dataset.dat'), 'w', encoding='utf-8') as wrt:
        for sample in samples:
            sample.write(wrt)

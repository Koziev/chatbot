# -*- coding: utf-8 -*-
"""
Подготовка таблиц замены для алгоритма смены грамматического лица в чатботе.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import io
import pickle
import sys
import argparse
import os

from utils.tokenizer import Tokenizer


parser = argparse.ArgumentParser(description='Grammatical dictionary preparation for person changer')
parser.add_argument('--run_mode', type=str, default='train', choices='train query'.split(), help='what to do')
parser.add_argument('--output_dir', type=str, default='../../tmp', help='directory to store results in')
parser.add_argument('--data_dir', type=str, default='../../data')

args = parser.parse_args()
run_mode = args.run_mode
output_dir = args.output_dir
data_dir = args.data_dir

if run_mode == 'train':
    word_1s = [set(), set(), set()]
    word_2s = [set(), set(), set()]

    # Предполагается, что в каталоге с данными data лежит файл с выгрузкой содержимого
    # грамматического словаря русского языка (https://github.com/Koziev/GrammarEngine)
    with io.open(os.path.join(data_dir, 'word2tags.dat'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            tx = line.strip().split(u'\t')
            if len(tx) == 5:
                tags = tx[3]
                if u'ЧИСЛО:ЕД' in tags:
                    if u'НАКЛОНЕНИЕ:ИЗЪЯВ' in tags:
                        incl = 0
                    elif u'НАКЛОНЕНИЕ:ПОБУД' in tags:
                        incl = 1
                    else:
                        incl = 2

                    if u'ЛИЦО:1' in tags:
                        word = tx[0].lower()
                        part_of_speech = tx[1]
                        word_1s[incl].add((word, part_of_speech))

                    if u'ЛИЦО:2' in tags:
                        word = tx[0].lower()
                        part_of_speech = tx[1]
                        word_2s[incl].add((word, part_of_speech))

    wordform2lemma = [dict(), dict(), dict()]
    lemma2word_1s = [dict(), dict(), dict()]
    lemma2word_2s = [dict(), dict(), dict()]
    with io.open(os.path.join(data_dir, 'word2lemma.dat'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            tx = line.strip().split(u'\t')
            word = tx[0].lower()
            lemma = tx[1].lower()
            part_of_speech = tx[2]

            k = (word, part_of_speech)
            for incl in range(3):
                if k in word_1s[incl]:
                    wordform2lemma[incl][k] = lemma
                    lemma2word_1s[incl][lemma] = k

                if k in word_2s[incl]:
                    wordform2lemma[incl][k] = lemma
                    lemma2word_2s[incl][lemma] = k

    person_change_1s_2s = dict()
    person_change_2s_1s = dict()

    lemmas = set()
    for incl in range(3):
        lemmas.update(lemma2word_2s[incl].keys())
        lemmas.update(lemma2word_1s[incl].keys())

    for lemma in lemmas:
        for incl in range(3):
            if lemma in lemma2word_1s[incl] and lemma in lemma2word_2s[incl]:
                p1s = lemma2word_1s[incl][lemma]
                p2s = lemma2word_2s[incl][lemma]
                person_change_1s_2s[p1s[0].lower()] = p2s[0].lower()
                person_change_2s_1s[p2s[0].lower()] = p1s[0].lower()

    # хардкод - где-то сверху баг
    person_change_1s_2s[u'меня'] = u'тебя'
    person_change_1s_2s[u'мне'] = u'тебе'
    person_change_1s_2s[u'мной'] = u'тобой'
    person_change_1s_2s[u'мною'] = u'тобою'
    person_change_1s_2s[u'я'] = u'ты'
    person_change_1s_2s[u'мое'] = u'твое'
    person_change_1s_2s[u'моя'] = u'твоя'
    person_change_1s_2s[u'мой'] = u'твой'
    person_change_1s_2s[u'мои'] = u'твои'
    person_change_1s_2s[u'моего'] = u'твоего'
    person_change_1s_2s[u'моей'] = u'твоей'
    person_change_1s_2s[u'моих'] = u'твоих'
    person_change_1s_2s[u'моим'] = u'твоим'
    person_change_1s_2s[u'моими'] = u'твоими'
    person_change_1s_2s[u'по-моему'] = u'по-твоему'

    person_change_2s_1s[u'тебя'] = u'меня'
    person_change_2s_1s[u'тебе'] = u'мне'
    person_change_2s_1s[u'тобой'] = u'мной'
    person_change_2s_1s[u'тобою'] = u'мною'
    person_change_2s_1s[u'ты'] = u'я'
    person_change_2s_1s[u'твое'] = u'мое'
    person_change_2s_1s[u'твоя'] = u'моя'
    person_change_2s_1s[u'твой'] = u'мой'
    person_change_2s_1s[u'твоего'] = u'моего'
    person_change_2s_1s[u'твоей'] = u'моей'
    person_change_2s_1s[u'твоих'] = u'моих'
    person_change_2s_1s[u'твоим'] = u'моим'
    person_change_2s_1s[u'твоими'] = u'моими'
    person_change_2s_1s[u'по-твоему'] = u'по-моему'


    # Сохраним все результаты в файле данных, чтобы чатбот мог воспользоваться ими без разбора
    # исходных словарей.
    model = dict()
    model['person_change_1s_2s'] = person_change_1s_2s
    model['person_change_2s_1s'] = person_change_2s_1s

    print(u'test person_change_1s_2s[меня] --> {}'.format(person_change_1s_2s[u'меня']) )
    print(u'test person_change_2s_1s[тебя] --> {}'.format(person_change_2s_1s[u'тебя']) )
    print(u'test person_change_2s_1s[мое] --> {}'.format(person_change_1s_2s[u'мое']) )
    print(u'test person_change_2s_1s[твое] --> {}'.format(person_change_2s_1s[u'твое']) )

    w1s = set()
    w2s = set()

    for incl in range(3):
        for z in word_1s[incl]:
            w1s.add(z[0])
        for z in word_2s[incl]:
            w2s.add(z[0])

    #model['word_1s'] = w1s
    #model['word_2s'] = w2s

    with open(os.path.join(output_dir, 'person_change_dictionary.pickle'), 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

if run_mode == 'query':
    with open(os.path.join(output_dir, 'person_change_dictionary.pickle'), 'rb') as f:
        model = pickle.load(f)

    w1s = model['word_1s']
    w2s = model['word_2s']
    person_change_1s_2s = model['person_change_1s_2s']
    person_change_2s_1s = model['person_change_2s_1s']

    tokenizer = Tokenizer()
    while True:
        print('\n')
        instr = raw_input('? ').decode(sys.stdout.encoding).strip().lower()
        if len(instr) == 0:
            break
        inwords = tokenizer.tokenize(instr)
        outwords = []
        for word in inwords:
            if word in w1s:
                outwords.append(person_change_1s_2s[word])
            elif word in w2s:
                outwords.append(person_change_2s_1s[word])
            else:
                outwords.append(word)

        print(u'{}'.format(u' '.join(outwords)))

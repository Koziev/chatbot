# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки валидатора синтаксиса (nn_syntax_validator.py).
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import io
import os
import numpy as np
import logging

import rutokenizer


def clean_input(s):
    s = s.replace('(-)', '').replace('(+)', '').replace('T:', '').replace('Q:', '').replace('A:', '').replace(u'ё', u'е')
    return s.strip().lower()


def clean_line(s):
    s = s.replace(u' ?', u'?').replace(u' ,', u',').replace(u' :', u',')\
        .replace(u' .', u'.').replace(u'( ', u'(').replace(u' )', u')')
    s = s.replace(u'ё', u'е')
    s = s[0].upper()+s[1:]
    return s


def remove_terminator(words):
    if words[0] == u'-':
        return remove_terminator(words[1:])
    else:
        if words[-1] in (u'.', u'?', u'!'):
            return words[:-1]
        else:
            return words


def load_samples(data_folder):
    logging.info('Loading samples...')
    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    samples = []
    emitted = set()

    with io.open(os.path.join(data_folder, 'invalid_syntax_dataset.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            if not line.startswith('#'):
                words = tokenizer.tokenize(line.strip().lower())
                if len(words) > 0:
                    words = remove_terminator(words)
                    key = u' '.join(words)
                    if key not in emitted:
                        samples.append((words, 0))
                        emitted.add(key)

    # В отдельном файле - валидные (но возможно не всегда разумные) сэмплы
    with io.open(os.path.join(data_folder, 'valid_syntax_dataset.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            if not line.startswith('#'):
                words = tokenizer.tokenize(line.strip().lower())
                if len(words) > 0:
                    words = remove_terminator(words)
                    key = u' '.join(words)
                    if key not in emitted:
                        samples.append((words, 1))
                        emitted.add(key)


    # Предполагаем, что корпус текстов для N-грамм содержит хорошие образцы
    with io.open(os.path.join(data_folder, 'ngrams_corpus.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            if not line.startswith('#'):
                words = tokenizer.tokenize(line.strip().lower())
                if len(words) > 0:
                    words = remove_terminator(words)
                    key = u' '.join(words)
                    if key not in emitted:
                        samples.append((words, 0))
                        emitted.add(key)

    for inpath in ['paraphrases.txt', 'pqa_all.dat']:
        with io.open(os.path.join(data_folder, inpath), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                if not line.startswith('#'):
                    s = clean_input(line)
                    words = tokenizer.tokenize(s)
                    if len(words) > 0:
                        if u'ты' in words:
                            words = remove_terminator(words)
                            key = u' '.join(words)
                            if key not in emitted:
                                emitted.add(key)
                                samples.append((words, 1))

    print('sample count={}'.format(len(samples)))

    nb0 = sum((label == 0) for (words, label) in samples)
    nb1 = sum((label == 1) for (words, label) in samples)
    print('nb0={} nb1={}'.format(nb0, nb1))

    max_wordseq_len = max(len(words) for (words, label) in samples)
    print('max_wordseq_len={}'.format(max_wordseq_len))

    return samples


if __name__ == '__main__':
    data_folder = '../../data'
    output_file = '../../data/syntax_validator_dataset.csv'

    samples = load_samples(data_folder)
    with io.open(output_file, 'w', encoding='utf-8') as wrt:
        wrt.write(u'sample\tlabel\n')
        for words, label in samples:
            wrt.write(u'{}\t{}\n'.format(u' '.join(words), label))

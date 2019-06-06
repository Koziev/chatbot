# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки entity extraction моделей
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import gensim
import numpy as np
import io
import argparse
import random
import logging
import collections

from utils.tokenizer import Tokenizer
import utils.logging_helpers


class Sample:
    def __init__(self, phrase, entity, value):
        self.phrase = phrase
        self.entity = entity
        self.value = value



def load_samples(input_path):
    print(u'Loading samples from {}'.format(input_path))

    # Для каждого класса извлекаемых сущностей получаем отдельный набор сэмплов
    entity2samples = dict()

    max_inputseq_len = 0
    tokenizer = Tokenizer()
    tokenizer.load()

    with io.open(input_path, 'r', encoding='utf-8') as rdr:
        current_entity = None
        for line in rdr:
            line = line.strip()
            if line and not line.startswith('#'):  # пропускаем комментарии и пустые строки
                if line.startswith('entity='):
                    current_entity = line.split('=')[1]
                    if current_entity not in entity2samples:
                        entity2samples[current_entity] = []
                else:
                    tx = line.split('|')
                    phrase = tx[0].strip()
                    value = u'' if len(tx) == 1 else tx[1].strip()

                    if phrase.endswith(u'.'):
                        phrase = phrase[:-1]
                    phrase = phrase.replace(u'?', u'').replace(u'!', u'')

                    phrase_tokens = tokenizer.tokenize(phrase)
                    value_tokens = tokenizer.tokenize(value)

                    sample = Sample(u' '.join(phrase_tokens), current_entity, u' '.join(value_tokens))
                    entity2samples[current_entity].append(sample)
                    max_inputseq_len = max(max_inputseq_len, len(phrase_tokens))

    print('max_inputseq_len={}'.format(max_inputseq_len))

    print('Count of samples per entity:')
    for entity, samples in sorted(entity2samples.items(), key=lambda samples: len(samples)):
        print(u'entity={} count={}'.format(entity, len(samples)))

    return entity2samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Подготовка датасета для entity extraction')
    parser.add_argument('--input', type=str, default='../../data/entity_extraction.txt', help='path to input dataset')
    parser.add_argument('--out_dir', type=str, default='../../data', help='папка для сохранения результатов')

    args = parser.parse_args()

    out_folder = args.out_dir
    input_path = args.input

    entity2samples = load_samples(input_path)

    res_path = os.path.join(out_folder, 'entities_dataset.tsv')
    with io.open(res_path, 'w', encoding='utf-8') as wrt:
        wrt.write(u'phrase\tentity\tvalue\n')
        for entity, samples in entity2samples.items():
            for sample in samples:
                wrt.write(u'{}\t{}\t{}\n'.format(sample.phrase, sample.entity, sample.value))

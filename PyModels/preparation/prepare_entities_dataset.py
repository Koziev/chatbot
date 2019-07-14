# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки entity extraction моделей для чатбота.
06.07.2019 экспорт датасета в формат Spacy
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import io
import argparse
import json

from utils.tokenizer import Tokenizer


MAX_ENTITY_PER_SAMPLE = 3


class Sample:
    def __init__(self, phrase, entity, value):
        self.phrase = phrase
        self.entity = entity
        self.value = value


def load_samples(input_path):
    print(u'Loading samples from {}'.format(input_path))

    # Для каждого класса извлекаемых сущностей получаем отдельный набор сэмплов
    entity2samples = dict()
    sample2entities = dict()

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
                    value = u' '.join(value_tokens)

                    sample = Sample(u' '.join(phrase_tokens), current_entity, value)
                    entity2samples[current_entity].append(sample)
                    max_inputseq_len = max(max_inputseq_len, len(phrase_tokens))

                    if sample.phrase not in sample2entities:
                        sample2entities[sample.phrase] = set()

                    sample2entities[sample.phrase].add((current_entity, value))

    print('max_inputseq_len={}'.format(max_inputseq_len))

    print('Count of samples per entity:')
    for entity, samples in sorted(entity2samples.items(), key=lambda samples: len(samples)):
        print(u'entity={} count={}'.format(entity, len(samples)))

    return entity2samples, sample2entities


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u'Подготовка датасета для entity extraction')
    parser.add_argument('--input', type=str, default='../../data/entity_extraction.txt', help=u'path to input dataset')
    parser.add_argument('--out_dir', type=str, default='../../data', help=u'папка для сохранения результатов')
    parser.add_argument('--tmp_dir', type=str, default='../../tmp', help=u'папка для отчетов и т.д.')

    args = parser.parse_args()

    out_folder = args.out_dir
    tmp_folder = args.tmp_dir
    input_path = args.input

    entity2samples, sample2entities = load_samples(input_path)

    res_path = os.path.join(out_folder, 'entities_dataset.tsv')
    err_path = os.path.join(tmp_folder, 'inconsistent_markup_in_entities_dataset.txt')

    with io.open(res_path, 'w', encoding='utf-8') as wrt,\
         io.open(err_path, 'w', encoding='utf-8') as err_wrt:
        wrt.write(u'phrase\tentity\tvalue\tvalue2\tvalue3\n')
        for phrase, entities in sample2entities.items():
            entity_types = set(entity for entity, value in entities)
            for entity_type in entity_types:
                values = []
                for entity, value in entities:
                    if entity == entity_type:
                        values.append(value)

                if u'' in values and len(values) > 1:
                    msg = u'Inconsistent markup for phrase "{}" and entity "{}"'.format(phrase, entity_type)
                    print(msg)
                    err_wrt.write(msg+u'\n')
                else:
                    good = True
                    for value1 in values:
                        for value2 in values:
                            if value1 != value2 and value1 in value2:
                                msg = u'Inconsistent markup for phrase "{}" and entity "{}"'.format(phrase, entity_type)
                                print(msg)
                                err_wrt.write(msg + u'\n')
                                good = False

                    if good:
                        if len(values) > MAX_ENTITY_PER_SAMPLE:
                            raise RuntimeError()
                        elif len(values) < MAX_ENTITY_PER_SAMPLE:
                            values.extend(u''*(MAX_ENTITY_PER_SAMPLE-len(values)))

                        wrt.write(u'{}\t{}\t{}\n'.format(phrase, entity_type, '\t'.join(values)))

    # Экспорт данных для обучения Spacy
    spacy_data = []
    for phrase, entities in sample2entities.items():
        padded_phrase = u' ' + phrase + u' '

        spacy_entities = []
        for entity, value in entities:
            padded_entity = u' ' + value + u' '
            if padded_entity in padded_phrase:
                pos1 = padded_phrase.index(padded_entity)
                pos2 = pos1 + len(padded_entity) - 2
                spacy_entities.append((pos1, pos2, entity))

        spacy_data.append((phrase, {'entities': spacy_entities}))

    res_path = os.path.join(out_folder, 'spacy_entities_dataset.json')
    with open(res_path, 'w') as f:
        json.dump(spacy_data, f, indent=4, encoding='utf-8')

    # НАЧАЛО ОТЛАДКИ
    # res_path = os.path.join(out_folder, 'spacy_entities_dataset.py')
    # with io.open(res_path, 'w', encoding='utf-8') as wrt:
    #     wrt.write(u'TRAIN_DATA=[\n')
    #     for sample, data in spacy_data:
    #         wrt.write(u"(u'{}', {{'entities': [({}, {}, 'date_time')]}}),\n".format(sample, data['entities'][0][0], data['entities'][0][1]))
    #     wrt.write(u']\n')
    # КОНЕЦ ОТЛАДКИ


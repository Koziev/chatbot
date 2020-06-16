# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировка модели определения необходимости интерпретации
(раскрытия анафоры, гэппинга и т.д.) в реплике пользователя.

Для проекта чатбота https://github.com/Koziev/chatbot

30-05-2020 Добавлена загрузка сэмплов из question interpretation датасета в общую кучу
"""

from __future__ import division
from __future__ import print_function

import io
import os
import random
import logging
import collections

import numpy as np
import pandas as pd
import csv

import ruchatbot.utils.logging_helpers


random.seed(123456789)
np.random.seed(123456789)


class Sample:
    def __init__(self, phrase, y, weight):
        assert(len(phrase) > 0)
        assert(y in [0, 1])
        self.phrase = phrase
        self.y = y
        self.weight = weight

    def __repr__(self):
        return '{} y={} w={}'.format(self.phrase, self.y, self.weight)


def remove_terminators(s):
    """ Убираем финальные пунктуаторы ! ? ."""
    return s[:-1].strip() if s[-1] in u'?!.' else s


def load_samples(input_path):
    samples0 = set()  # не нуждаются в интерпретации
    samples1 = collections.Counter()  # нуждаются в интерпретации
    with io.open(input_path, 'r', encoding='utf-8') as rdr:
        phrases = []
        for iline, line in enumerate(rdr):
            line = line.strip()
            if line.startswith('#'):  # комментарии пропускаем
                continue

            if len(line) == 0:
                if len(phrases) > 0:
                    # Из набора берем первую строку и последнюю.

                    # Первая строка у нас всегда содержит полную фразу.
                    if len(phrases) > 1:
                        phrase = phrases[0]
                        if '|' not in phrase:
                            samples0.add(remove_terminators(phrase.strip()))

                    # Последняя строка должна содержать краткую реплика и полную, разделенные символом |
                    phrase = phrases[-1]
                    if '|' in phrase:
                        px = phrase.lower().split('|')
                        if px[0] != px[1]:
                            left = remove_terminators(px[0].strip())
                            if not left:
                                print(u'Empty left part in line #{} in "{}"'.format(iline, input_path))
                                exit(0)
                            else:
                                samples1[left] += 1

                        right = remove_terminators(px[1].strip())
                        if not right:
                            print(u'Empty right part in line #{}'.format(iline))
                            exit(0)
                        else:
                            # правая часть интерпретации не нуждается в переинтерпретации,
                            # так как совпадает с левой.
                            samples0.add(right)
                    else:
                        samples0.add(phrase)

                phrases = []
            else:
                if line.startswith(u'#'):  # строки комментариев
                    continue
                else:
                    phrases.append(line)

    return list(samples0), list(samples1.items())


if __name__ == '__main__':
    data_folder = '../../data'
    tmp_folder = '../../tmp'

    # настраиваем логирование в файл
    ruchatbot.utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'prepare_req_interpretation_classif.log'))

    logging.info('Start "prepare_req_interpretation_classif.py"')

    samples = []

    # Ручной датасет №1.
    samples1_0, samples1_1 = load_samples(os.path.join(data_folder, 'interpretation.txt'))
    samples.extend(Sample(sample, 0, weight=1) for sample in samples1_0)
    samples.extend(Sample(sample, 1, weight=weight) for sample, weight in samples1_1)

    # Ручной датасет с интерпретацией вопросов
    samples1_0, samples1_1 = load_samples(os.path.join(data_folder, 'question_interpretation.txt'))
    samples.extend(Sample(sample, 0, weight=1) for sample in samples1_0)
    samples.extend(Sample(sample, 1, weight=weight) for sample, weight in samples1_1)

    # Из автоматического датасета возьмем столько же сэмплов, сколько получилось
    # из ручного датасета.
    samples2_0, samples2_1 = load_samples(os.path.join(data_folder, 'interpretation_auto_5.txt'))
    samples2_1 = np.random.permutation(samples2_1)

    # оставим примерно столько автосэмплов, сколько извлечено из ручного датасета
    samples2_1 = samples2_1[:len(samples1_1) * 2]

    samples.extend(Sample(sample, 0, weight=1) for sample in samples2_0)
    samples.extend(Sample(sample, 1, weight=weight) for sample, weight in samples2_1)

    all_texts = set(sample.phrase for sample in samples)

    # Из таблицы трансляции приказов возьмем все строки - они не требуют интерпретации
    with io.open(os.path.join(data_folder, 'orders.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s and s not in all_texts:
                samples.append(Sample(s, 0, weight=1))
                all_texts.add(s)

    # Из демо-FAQ возьмем вопросы, они тоже гарантированно полные
    with io.open(os.path.join(data_folder, 'faq2.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s and s.startswith(u'Q: '):
                s = s.replace(u'Q:', u'').strip()
                if s not in all_texts:
                    samples.append(Sample(s, 0, weight=1))
                    all_texts.add(s)


    # Добавляем негативные примеры, то есть такие предложения, для которых
    # не надо выполнять интерпретацию.
    df = pd.read_csv(os.path.join(data_folder, 'premise_question_relevancy.csv'),
                     encoding='utf-8',
                     delimiter='\t',
                     quoting=3)

    for premise in set(df.premise.values[:len(samples1_1)]):
        if premise not in all_texts:
            samples.append(Sample(premise, 0, weight=1))
            all_texts.add(premise)

    for question in set(df.question.values):
        if question not in all_texts:
            samples.append(Sample(question, 0, weight=1))
            all_texts.add(question)

    samples = np.random.permutation(samples)

    nb_0 = sum(sample.y == 0 for sample in samples)
    nb_1 = sum(sample.y == 1 for sample in samples)
    logging.info('nb_0=%d nb_1=%d', nb_0, nb_1)

    df = pd.DataFrame(columns='text label'.split(), index=None)
    df['text'] = [sample.phrase for sample in samples]
    df['label'] = [sample.y for sample in samples]
    df['weight'] = [sample.weight for sample in samples]

    outpath = os.path.join(data_folder, 'req_interpretation_dataset.csv')
    logging.info('Writing dataset with shape={} rows to "{}"'.format(df.shape, outpath))
    df.to_csv(outpath, sep='\t', header=True, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)

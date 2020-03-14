# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировка модели генерации реплики бота.

Для проекта чатбота https://github.com/Koziev/chatbot
"""

from __future__ import division
from __future__ import print_function

import io
import os
import random
import logging

import numpy as np
import pandas as pd
import csv

import ruchatbot.utils.logging_helpers

tmp_folder = '../../tmp'
data_folder = '../../data'


def normalize_line(s):
    s = s.replace(u'\u00A0', u' ').strip()
    if s.startswith(u'-'):
        return s[1:].strip()
    else:
        return s


left_texts = []
right_texts = []

if False:
    # Ручной датасет с примерами реплик
    nb_handmade = 0
    #fn = 'replica_generator_handmade_dataset.txt'
    fn = 'smalltalk.txt'
    with io.open(os.path.join(data_folder, fn), 'r', encoding='utf-8') as rdr:
        q_list = []
        a_list = []
        for iline, line in enumerate(rdr):
            line = line.strip()
            if len(line) == 0:
                for q in q_list:
                    for a in a_list:
                        a = a.strip()
                        left_texts.append(q)
                        right_texts.append(a)
                        nb_handmade += 1

                q_list = []
                a_list = []
            elif line.startswith('Q:'):
                q_list.append(line.replace('Q:', '').strip())
            elif line.startswith('A:'):
                a_list.append(line.replace('A:', '').strip())
    print('{} samples from replica_generator_handmade_dataset.txt have been added.'.format(nb_handmade))

if False:
    with io.open('/home/inkoziev/polygon/NLPContests/Yandex2018/data/train.tsv', 'r', encoding='utf-8') as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            group_id = int(parts[0])
            context = u''
            for i, part in enumerate(parts):
                if i > 0:
                    if len(part) == 1 and part[0].isdigit():
                        reply = parts[i+1]
                        label = parts[i+2]
                        if label == 'bad':
                            label = 0
                        elif label == 'neutral':
                            label = 1
                        elif label == 'good':
                            label = 2
                        else:
                            raise RuntimeError()

                        label_w = float(parts[i+3])

                        if label == 2:
                            left_texts.append(context)
                            right_texts.append(reply)

                        break
                    else:
                        if len(context) > 0:
                            context += u' '
                        context += part

if True:
    nb_dialogues = 0
    with io.open(os.path.join(tmp_folder, 'dialogues.txt'), 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            s = line.strip()
            if s:
                lines.append(s)
            else:
                if len(lines) > 1:
                    left_text = u' '.join(normalize_line(line) for line in lines[:-1])
                    right_text = normalize_line(lines[-1])
                    left_texts.append(left_text)
                    right_texts.append(right_text)
                    nb_dialogues += 1
                lines = []
                if len(left_texts) > 1000000:
                    break  # ограничиваем кол-во диалогов
    print('{} samples from dialogues.txt have been added'.format(nb_dialogues))

df = pd.DataFrame(columns='context response'.split(), index=None)
df['context'] = left_texts
df['response'] = right_texts

outpath = os.path.join(data_folder, 'replica_generator_dataset.csv')
logging.info('Writing dataset with shape={} rows to {}'.format(df.shape, outpath))
df.to_csv(outpath, sep='\t', header=True, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)

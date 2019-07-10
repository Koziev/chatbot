# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки модели определения intent'а в NLU движке чатбота
27.06.2019 - формат датасета приведен в соответствие с RASA
"""

from __future__ import division  # for python2 compatability
from __future__ import print_function

import io
import collections
import operator
import pandas as pd
import csv


input_path = '../../data/intents.txt'
output_path = '../../data/intents_dataset.csv'
samples = set()

with io.open(input_path, 'r', encoding='utf-8') as rdr:
    current_intent = None
    for iline, line in enumerate(rdr):
        if line.startswith('#'):
            if line.startswith('##'):
                if 'intent:' in line:
                    current_intent = line.split(':')[1].strip()
                else:
                    raise RuntimeError()
            else:
                # комментарии пропускаем
                continue
        else:
            line = line.strip()
            if line.startswith('-'):  # в файлах RASA строки начинаются с -
                line = line[1:]
            if line:
                if current_intent:
                    samples.add((line, current_intent))
                else:
                    print('line #{}: Current intent is "None"!'.format(iline))
                    exit(0)

samples = list(samples)

print('Intent frequencies:')
intent2freq = collections.Counter()
intent2freq.update(map(operator.itemgetter(1), samples))
for intent, freq in intent2freq.most_common():
    print(u'{}\t{}'.format(intent, freq))

print('{} samples, {} intents in result dataset'.format(len(samples), len(intent2freq)))
df = pd.DataFrame(columns='phrase intent'.split(), index=None)
for sample in samples:
    df = df.append({'phrase': sample[0], 'intent': sample[1]}, ignore_index=True)

df.to_csv(output_path, sep='\t', header=True, index=False,
          encoding='utf-8', compression=None, quoting=csv.QUOTE_NONE, quotechar=None,
          line_terminator='\n', chunksize=None, tupleize_cols=None, date_format=None,
          doublequote=True,
          escapechar=None, decimal='.')

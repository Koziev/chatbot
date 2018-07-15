# -*- coding: utf-8 -*-
'''
Из файла с тройками premise-question-answer берем предпосылки и вопросы
и сохраняем их в отдельных текстовых файлах для последующих
экспериментов с разными метриками релевантности.

(c) by Koziev Ilya inkoziev@gmail.com
'''

from __future__ import division  # for python2 compatability
from __future__ import print_function

import codecs
import os

import pandas as pd

from utils.tokenizer import Tokenizer

tmp_folder = '../tmp'
data_folder = '../data'
qa_path = '../data/qa.txt'

# ---------------------------------------------------------------

tokenizer = Tokenizer()
wrt_premises = codecs.open(os.path.join(tmp_folder, 'premises.txt'), 'w', 'utf-8')
wrt_questions = codecs.open(os.path.join(tmp_folder, 'questions.txt'), 'w', 'utf-8')

with codecs.open(qa_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        line = line.strip()
        if line.startswith(u'T:'):
            words = tokenizer.tokenize(line.replace(u'T:', u'').strip())
            wrt_premises.write(u'{}\n'.format(u' '.join(words)))
        elif line.startswith(u'Q:'):
            words = tokenizer.tokenize(line.replace(u'Q:', u'').strip())
            wrt_questions.write(u'{}\n'.format(u' '.join(words)))

wrt_premises.close()
wrt_questions.close()

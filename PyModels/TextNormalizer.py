# -*- coding: utf-8 -*-
# coding: utf-8
"""
Функции для нормализации текста - уборка мусора, замена сокращений и т.д.
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability

import pandas as pd
import collections
from collections import Counter
import tqdm
import re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import functools
import sys
import codecs
from nltk.stem.snowball import RussianStemmer, EnglishStemmer
from Tokenizer import Tokenizer
import Abbrev


def preprocess_line(text0):
    text = re.sub(u'\\[[0-9]+\\]', u'', text0)
    return text

# -----------------------------------------------------------

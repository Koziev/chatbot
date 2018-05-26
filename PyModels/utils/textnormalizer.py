# -*- coding: utf-8 -*-
# coding: utf-8
"""
Функции для нормализации текста - уборка мусора, замена сокращений и т.д.
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability

import re


def preprocess_line(text0):
    text = re.sub(u'\\[[0-9]+\\]', u'', text0)
    return text

# -----------------------------------------------------------

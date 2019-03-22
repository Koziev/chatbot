# -*- coding: utf-8 -*-
'''
Генерация синтетических паттернов по шаблону "Сейчас HH часов MM минут"+"Сколько сейчас времени"
'''

from __future__ import print_function
from __future__ import division  # for python2 compatibility

import codecs
import os

with codecs.open('../data/current_time_pqa.txt', 'w', 'utf-8') as wrt:
    for h in range(1,25):
        for m in range(0,60):
            timestr = u'{}'.format(h)
            hh = h%10
            if hh==1:
                timestr += u' час'
            elif 2<=hh<=4:
                timestr += u' часа'
            else:
                timestr += u' часов'

            timestr += u' {}'.format(m)
            mm = m%10
            if mm==1:
                timestr += u' минута'
            elif 2<=mm<=4:
                timestr += u' минуты'
            else:
                timestr += u' минут'

            premise = u'Сейчас '+timestr
            answer = timestr

            wrt.write(u'T: '+premise+u'\n')
            wrt.write(u'Q: Сколько сейчас времени?\n')
            wrt.write(u'A: '+answer+u'\n')
            wrt.write('\n')
            wrt.write(u'T: '+premise+u'\n')
            wrt.write(u'Q: Сейчас сколько времени?\n')
            wrt.write(u'A: '+answer+u'\n')
            wrt.write('\n')

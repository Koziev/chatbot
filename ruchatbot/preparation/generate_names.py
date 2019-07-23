# -*- coding: utf-8 -*-
'''
Генерация синтетических паттернов по шаблону "Меня зовут XXX"+"Как меня зовут"
'''

from __future__ import print_function
from __future__ import division  # for python2 compatibility

import codecs
import os

names = [ u'Михаил', u'Кирил', u'Петр', u'Миша', u'Петя', u'Иван', u'Андрей', u'Андрюша', u'Алексей',
          u'Саша', u'Константин', u'Костя', u'Василий', u'Вася', u'Федор', u'Федя', u'Николай', u'Коля',
          u'Прохор', u'Иннокентий', u'Кеша', u'Илья', u'Семен', u'Степан', u'Степа', u'Дмитрий', u'Дима',
          u'Витя', u'Борис', u'Боря', u'Игорь', u'Гоша', u'Виталий', u'Виталя', u'Виталик',
          u'Владимир', u'Володя', u'Арсений', u'Сергей', u'Сережа', u'Валерий', u'Валера',
          u'Марк', u'Геннадий', u'Гена', u'Егор', u'Жора', u'Захар', u'Руслан', u'Юлий', u'Эдуард',
        ]

with codecs.open('../data/names_pqa.txt', 'w', 'utf-8') as wrt:
    for name in names:
        answer = name
        for q in [u'меня', u'тебя', u'его']:

            premise = q + u' зовут '+name

            wrt.write(u'T: '+premise+u'\n')
            wrt.write(u'Q: {} как зовут?\n'.format(q))
            wrt.write(u'A: '+answer+u'\n')
            wrt.write('\n')
            wrt.write(u'T: '+premise+u'\n')
            wrt.write(u'Q: Как {} зовут?\n'.format(q))
            wrt.write(u'A: '+answer+u'\n')
            wrt.write('\n')
            wrt.write(u'T: '+premise+u'\n')
            wrt.write(u'Q: Зовут {} как?\n'.format(q))
            wrt.write(u'A: '+answer+u'\n')
            wrt.write('\n')

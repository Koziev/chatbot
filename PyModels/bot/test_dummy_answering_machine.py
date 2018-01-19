# -*- coding: utf-8 -*-
'''
'''

from __future__ import print_function
from __future__ import division  # for python2 compatability
import sys
from dummy_answering_machine import DummyAnsweringMachine


bot = DummyAnsweringMachine()

while True:
    print('\n')
    question = raw_input('Q:> ').decode(sys.stdout.encoding).strip().lower()
    if len(question)==0:
        break
    bot.push_phrase('test', question)
    while True:
        answer = bot.pop_phrase('test')
        if len(answer)==0:
            break

        print(answer)



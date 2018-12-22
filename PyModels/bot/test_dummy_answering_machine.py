# -*- coding: utf-8 -*-

from __future__ import print_function

from dummy_answering_machine import DummyAnsweringMachine
from console_utils import input_kbd


bot = DummyAnsweringMachine()

while True:
    print('\n')
    question = input_kbd('Q:> ').strip().lower()
    if len(question) == 0:
        break
    bot.push_phrase('test', question)
    while True:
        answer = bot.pop_phrase('test')
        if len(answer) == 0:
            break

        print(answer)

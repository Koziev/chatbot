# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для тестирования экземпляра FAQ бота, создаваемого
функцией ruchatbot.create_qa_bot
"""

import logging

import ruchatbot

profile_path = '../data/profile_1.json'
models_folder = '../tmp'
data_folder = '../data'
w2v_folder = '../tmp'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

bot = ruchatbot.create_qa_bot(profile_path, models_folder, data_folder, w2v_folder, debugging=True)

user_id = 'some_user'

while True:
    while True:
        answer = bot.pop_phrase(user_id)
        if len(answer) == 0:
            break

        print(u'B:> {}'.format(answer))

    question = input('H:>').strip()
    if len(question) > 0:
        bot.push_phrase(user_id, question)

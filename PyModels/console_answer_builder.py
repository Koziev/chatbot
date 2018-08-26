# -*- coding: utf-8 -*-
"""
Консольный фронтэнд для простого генератора ответов - бота без хранилища фактов,
который получает от тестера одну предпосылку и вопрос, и генерирует ответ.
Для ручного тестирования моделей.
Часть проекта вопросно-ответной системы https://github.com/Koziev/chatbot 
"""

import os
import logging
import argparse

from bot.text_utils import TextUtils
from bot.console_utils import input_kbd, print_answer, print_error
from bot.answer_builder import AnswerBuilder
from bot.word_embeddings import WordEmbeddings


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.ERROR)

parser = argparse.ArgumentParser(description='answer builder console tester')
parser.add_argument('--models_folder', type=str, default='../tmp', help='path to folder with pretrained models')
parser.add_argument('--data_folder', type=str, default='../data', help='path to folder with dictionaries')
parser.add_argument('--w2v_folder', type=str, default='../data', help='path to folder with word2vector models')

args = parser.parse_args()
models_folder = args.models_folder
data_folder = args.data_folder
w2v_folder = args.w2v_folder

text_utils = TextUtils()
text_utils.load_dictionaries(data_folder)
bot = AnswerBuilder()
bot.load_models(models_folder)

word_embeddings = WordEmbeddings()
word_embeddings.load_wc2v_model(os.path.join(models_folder, 'wordchar2vector.dat'))
for p in bot.get_w2v_paths():
    word_embeddings.load_w2v_model(os.path.join(w2v_folder, os.path.basename(p)))

max_nb_premises = 1

while True:
    print('\n')
    phrases = []
    while True:
        phrase = input_kbd(u':> ')
        if len(phrase) > 0:
            phrases.append(phrase)
            if phrase[-1] == u'?':
                break
            elif len(phrases) == max_nb_premises+1:
                break
        elif len(phrases) > 1:
            break

    if len(phrases) < 2:
        print_error('At least 2 phrases expected!')
        continue

    premises = phrases[:-1]
    question = phrases[-1]

    answer = bot.build_answer_text(premises[0], question, text_utils, word_embeddings)
    print_answer(u'A:>', answer)

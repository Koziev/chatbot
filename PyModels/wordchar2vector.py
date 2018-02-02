# -*- coding: utf-8 -*-
"""
Тренировка модели для превращения символьной цепочки слова в вектор.
RNN и CNN варианты энкодера. Реализовано на Keras.

Список слов, для которых строится модель, читается из файла ../tmp/known_words.txt
и должен быть предварительно сформирован скриптом prepare_wordchar_dataset.py
"""

from __future__ import print_function

__author__ = "Ilya Koziev"

import argparse

from trainers.wordchar2vector_trainer import Wordchar2Vector_Trainer

parser = argparse.ArgumentParser(description='Training the wordchar2vector embeddings for words')
parser.add_argument('--input_file', default='../tmp/known_words.txt', help='input text file with words to be processed')
parser.add_argument('--out_file', default='../tmp/wordchar2vector.dat', help='output text file containing with word vectors in word2vec text format')
parser.add_argument('--model_dir', help='folder with model files', default='../tmp')
parser.add_argument('--train', default=1, type=int)
parser.add_argument('--vectorize', default=0, type=int)
parser.add_argument('--dims', default=56, type=int)
parser.add_argument('--char_dims', default=0, type=int)
parser.add_argument('--tunable_char_embeddings', default=0, type=int)
parser.add_argument('--arch_type', default='rnn', type=str)
parser.add_argument('--batch_size', default=250, type=int)

args = parser.parse_args()

model_dir = args.model_dir  # каталог для файлов модели - при тренировке туда записываются, при векторизации - оттуда загружаются
input_path = args.input_file  # из этого файла прочитаем список слов, на которых учится модель
out_file = args.out_file  # в этот файл будет сохранены векторы слов в word2vec-совместимом формате
do_train = args.train  # тренировать ли модель с нуля
do_vectorize = args.vectorize  # векторизовать ли входной список слов
vec_size = args.dims  # размер вектора представления слова для тренировки модели
char_dims = args.char_dims  # если векторы символов будут меняться при тренировке, то явно надо задавать размерность векторов символов
batch_size = args.batch_size  # размер минибатчей существенно влияет на точность, поэтому разрешаем задавать его
tunable_char_embeddings = args.tunable_char_embeddings  # делать ли настраиваемые векторы символов (True) или 1-hot (False)

# архитектура модели:
# cnn - сверточный энкодер
# rnn - рекуррентный энкодер
# lstm+cnn - гибридная сетка с параллельными рекуррентными и сверточными слоями
# cnn*lstm - сверточные слои и поверх них рекуррентные.
arch_type = args.arch_type

# -------------------------------------------------------------------

# Для упрощения отладки без задания параметров запуска - запросим с консоли
# выбор режима - перетренировка модели с нуля или только векторизация с использованием
# ранее натренированной модели.
if not do_train and not do_vectorize:
    while True:
        a1 = raw_input('0-train model\n1-calculate embeddings using pretrained model\n[0/1]: ')
        if a1 == '0':
            do_train = True
            do_vectorize = True
            print('Training the model...')
            break
        elif a1 == '1':
            do_train = False
            do_vectorize = True
            print('Calculating the word embeddings...')
            break
        else:
            print('Unrecognized choice "{}"'.format(a1))

# ---------------------------------------------------------------

trainer = Wordchar2Vector_Trainer(arch_type,
                                  tunable_char_embeddings,
                                  char_dims,
                                  model_dir,
                                  vec_size,
                                  batch_size)

if do_train:
    trainer.train(input_path)

if do_vectorize:
    trainer.vectorize(input_path, out_file)

print('\nDone.')

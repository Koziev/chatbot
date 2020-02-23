# -*- coding: utf-8 -*-
"""
Сжимаем модель w2v, оставляем только используемые в тренировочном датасете слова.
Также сохраняем модели в формате KeyedVectors для оптимизации времени загрузки
в чатботе через KeyedVectors.load(.., mmap='r')

Результат сохраняем в tmp.
"""

from __future__ import division
from __future__ import print_function

import os
import io
import gensim

tmp_dir = '../../tmp'

w2v_path = os.path.expanduser('~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin')

wc2v_src_path = os.path.join(tmp_dir, 'wordchar2vector.dat')
wc2v_res_path = os.path.join(tmp_dir, '../tmp/wc2v.kv')


print(u'Loading w2v from {}'.format(w2v_path))
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
w2v_dims = len(w2v.vectors[0])

all_words = set()
with io.open(os.path.join(tmp_dir, 'dataset_words.txt'), 'r', encoding='utf-8') as rdr:
    for line in rdr:
        word = line.strip()
        if word in w2v:
            all_words.add(word)

nb_words = len(all_words)
print('{} words'.format(nb_words))


print('Writing w2v text model...')
with io.open(os.path.join(tmp_dir, 'w2v_distill.tmp'), 'w', encoding='utf-8') as wrt:
    wrt.write(u'{} {}\n'.format(nb_words, w2v_dims))

    for word in all_words:
        word_vect = w2v[word]
        wrt.write(u'{} {}\n'.format(word, u' '.join([str(x) for x in word_vect])))

del w2v

w2v = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(tmp_dir, 'w2v_distill.tmp'), binary=False)

dest_path = os.path.join(tmp_dir, 'w2v.CBOW=1_WIN=5_DIM=64.bin')
print('Storing binary w2v model to "{}"...'.format(dest_path))
w2v.save_word2vec_format(dest_path, binary=True)

# Сохраняем в формате для быстрой загрузки, будет 2 файла.
w2v.save(os.path.join(tmp_dir, 'w2v.kv'))

print('Loading wc2v model from "{}"...'.format(wc2v_src_path))
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wc2v_src_path, binary=False)

print('Storing wc2v as KeyedVectors "{}"'.format(wc2v_res_path))
wc2v.save(wc2v_res_path)

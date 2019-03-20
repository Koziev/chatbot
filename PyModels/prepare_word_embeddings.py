# -*- coding: utf-8 -*-
"""
Сжимаем модель w2v, оставляем только используемые в тренировочном датасете слова.
Результат сохраняем в tmp.
"""

from __future__ import division
from __future__ import print_function

import os
import io
import gensim

tmp_dir = '../tmp'

w2v_path = os.path.expanduser('~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin')
print(u'Loading w2v from {}'.format(w2v_path))
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
w2v_dims = len(w2v.syn0[0])

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

print('Storing binary w2v model...')
w2v.wv.save_word2vec_format(os.path.join(tmp_dir, 'w2v.CBOW=1_WIN=5_DIM=64.bin'), binary=True)

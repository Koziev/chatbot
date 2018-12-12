# -*- coding: utf-8 -*-
'''
Готовим урезанные модели w2v, fasttext etc, в которых хранятся вектора только
для слов из списка dataset_words.txt (создается в prepare_wordchar_datset.py).
Это ускоряет загрузку моделей при экспериментах.
'''

from __future__ import print_function
from __future__ import division  # for python2 compatability

import os
import codecs
import gensim

dataset_words_path = '../tmp/dataset_words.txt'
output_folder = '../tmp'
input_folder = '/home/eek/polygon/w2v'

dataset_words = [u'_num_', u'</s>']
for line in codecs.open(dataset_words_path, 'r', 'utf-8'):
    word = line.strip()
    if len(word) > 0:
        dataset_words.append(word)

print('dataset_words.count={}'.format(len(dataset_words)))


if False:
    # --- FastTest ---
    for ft_path in ['fasttext.CBOW=1_WIN=5_DIM=32',
                    'fasttext.CBOW=1_WIN=5_DIM=64']:
        path = os.path.join(input_folder, ft_path)
        print('Loading fasttext model {}'.format(path))
        w2v = gensim.models.wrappers.FastText.load_fasttext_format(path)
        w2v_dim = w2v.vector_size
        print('w2v_dim={0}'.format(w2v_dim))

        model_name = os.path.basename(ft_path) + '.txt'
        new_path = os.path.join(output_folder, model_name)

        model_data = []
        for word in dataset_words:
            if word in w2v:
                model_data.append((word, w2v[word]))

        nb_words = len(model_data)
        print('Writing {} vectors to {}'.format(nb_words, new_path))
        with codecs.open(new_path, 'w', 'utf-8') as wrt:
            wrt.write('{} {}\n'.format(nb_words, w2v_dim))
            for word, word_vect in model_data:
                wrt.write(u'{} {}\n'.format(word, u' '.join([str(x) for x in word_vect])))


if False:
    # --- word2vec ---
    for w2v_path in ['w2v.CBOW=1_WIN=5_DIM=8.bin',
                     'w2v.CBOW=0_WIN=5_DIM=8.bin',
                     'w2v.CBOW=1_WIN=5_DIM=32.bin',
                     'w2v.CBOW=1_WIN=5_DIM=64.bin',
                     'w2v.CBOW=0_WIN=5_DIM=32.bin']:
        path = os.path.join(input_folder, w2v_path)
        print('Loading the w2v model {}'.format(path))
        w2v = gensim.models.KeyedVectors.load_word2vec_format(path, binary=not w2v_path.endswith('.txt'))
        w2v_dim = len(w2v.syn0[0])
        print('w2v_dim={0}'.format(w2v_dim))

        model_name = os.path.splitext(os.path.basename(w2v_path))[0] + '.txt'
        new_path = os.path.join(output_folder, model_name)

        model_data = []
        for word in dataset_words:
            if word in w2v:
                model_data.append((word, w2v[word]))

        nb_words = len(model_data)
        print('Writing {} vectors to {}'.format(nb_words, new_path))
        with codecs.open(new_path, 'w', 'utf-8') as wrt:
            wrt.write('{} {}\n'.format(nb_words, w2v_dim))
            for word, word_vect in model_data:
                wrt.write(u'{} {}\n'.format(word, u' '.join([str(x) for x in word_vect])))

# -*- coding: utf-8 -*-
"""
Кластеризация предложений после перевода их в векторную форму с помощью ранее
натренированной модели sent2vec (см. программу nn_relevancy.py)
"""

from __future__ import division
from __future__ import print_function

import codecs
import gc
import itertools
import json
import os
import sys
import argparse

import gensim
import numpy as np
import pandas as pd
import tqdm

import sklearn.cluster
import scipy.spatial.distance

import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.merge import concatenate, add, multiply
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer
from utils.segmenter import Segmenter


# Папка, куда программа nn_relevancy.py сохранила файлы моделей
model_folder = '../tmp'

# Путь к текстовому файлу, содержащему предложения для векторизации
# и последующей кластеризации.
input_path = '../data/facts6.txt'

nb_clusters = 100

# Путь к файлу, куда мы сохраним результаты кластеризации.
res_path = '../tmp/sent2vector.clusters.txt'


def vectorize_words(words, X_data, irow, word2vec):
    for iword, word in enumerate(words):
        if word in word2vec:
            X_data[irow, iword, :] = word2vec[word]





# некоторые необходимые для векторизации предложений параметры
# сохранены программой nn_relevancy.py в файле relevancy_model.config
with open(os.path.join(model_folder,'nn_relevancy_model.config'), 'r') as f:
    model_config = json.load(f)

    max_wordseq_len = model_config['max_wordseq_len']
    word2vector_path = model_config['w2v_path']
    wordchar2vector_path = model_config['wordchar2vector_path']
    PAD_WORD = model_config['PAD_WORD']
    word_dims = int(model_config['word_dims'])

# Загружаем модель sent2vec
arch_filepath2 = os.path.join(model_folder, 'sent2vector.arch')
weights_path2 = os.path.join(model_folder, 'sent2vector.weights')

with open(arch_filepath2, 'r') as f:
    model = model_from_json(f.read())

model.load_weights(weights_path2)


print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])
print('wc2v_dims={0}'.format(wc2v_dims))

print( 'Loading the w2v model {}'.format(word2vector_path) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
w2v_dims = len(w2v.syn0[0])
print('w2v_dims={0}'.format(w2v_dims))

word2vec = dict()
for word in wc2v.vocab:
    v = np.zeros( word_dims )
    v[w2v_dims:] = wc2v[word]
    if word in w2v:
        v[:w2v_dims] = w2v[word]

    word2vec[word] = v

del w2v
del wc2v
gc.collect()


# Загружаем список предложений
print(u'Loading sentences from {}...'.format(input_path))
sents = []  # здесь сохраним пары (строка_предложения, список_слов_предложения)
tokenizer = Tokenizer()
all_words = set()
with codecs.open(input_path, 'r', 'utf-8') as rdr:
    for line in rdr:
        sent = line.strip()
        if len(sent)>0:
            words = tokenizer.tokenize(sent)

            all_words_found = all( (word in word2vec) for word in word )
            if all_words_found:
                sents.append((sent, words))
                all_words.update(words)
nb_sents = len(sents)
print('{} sentences, {} distinct words in list'.format(nb_sents, len(all_words)))

# Будем заполнять входную матрицу
X_data = np.zeros((nb_sents, max_wordseq_len, word_dims), dtype=np.float32)
for isent, sent in enumerate(sents):
    words = sent[1]
    vectorize_words(words, X_data, isent, word2vec)

# Прогоняем через модель.
# Получаем вектор для каждого предложения
print('Running sent2vector model...')
sent_vect = model.predict(X_data)

# Выполняем кластеризацию.
print('Start k-means for {0} vectors, {1} clusters...'.format(nb_sents, nb_clusters))
#(codebook,labels) = scipy.cluster.vq.kmeans2( data=sent_vect, k=n_cluster )
kmeans = sklearn.cluster.KMeans(n_clusters=nb_clusters, max_iter=20, verbose=1, copy_x=False, n_jobs=1, algorithm='auto')
kmeans.fit(sent_vect)
labels = kmeans.labels_
codebook = kmeans.cluster_centers_
print('Finished.')

print('Printing clusters to {0}...'.format(res_path))
sent_vec_list = sent_vect
with codecs.open(res_path, 'w', 'utf-8') as wrt:
    for target_cluster in range(nb_clusters):
        print('{0}/{1}'.format(target_cluster, nb_clusters), end='\r')
        sys.stdout.flush()

        cluster_coord = codebook[target_cluster]

        sent_in_cluster = [(sents[isent][0], sent_vec_list[isent]) for (isent, l) in enumerate(labels) if l == target_cluster]
        #sent_in_cluster = sorted(sent_in_cluster, key=lambda z: -scipy.spatial.distance.cosine(cluster_coord, z[1]))
        sent_in_cluster = sorted(sent_in_cluster, key=lambda z: -scipy.spatial.distance.euclidean(cluster_coord, z[1]))
        sent_in_cluster = sent_in_cluster[: min(50, len(sent_in_cluster))]
        sent_in_cluster = [(s, scipy.spatial.distance.cosine(cluster_coord, v))  for (s, v) in sent_in_cluster]

        wrt.write('\n\ncluster #{}\n'.format(target_cluster))
        for sent, cos_sim in sent_in_cluster:
            wrt.write(u'{}\t{}\n'.format(cos_sim, sent))

print('All done.')

# -*- coding: utf-8 -*-
'''
Интерактивная проверка обученного Part-of-Speech Tagger'а (см. train_postagger.py)
Для чатбота https://github.com/Koziev/chatbot.
'''

from __future__ import print_function
from __future__ import division  # for python2 compatibility

import codecs
import os
import gc
import itertools
import numpy as np
import platform
import gensim
import json
import pycrfsuite
import utils.console_helpers


BEG_TOKEN = '<beg>'
END_TOKEN = '<end>'

gren_path = '../../data/word2tags.dat'


def normalize_word(word):
    return word.replace(' - ', '-').replace(u'ё', u'е').lower()


def get_word_features(word, prefix, word2vec, word2tags):
    features = dict()

    if use_gren and word in word2tags:
        for tag in word2tags[word]:
            features[u'tag[{}]={}'.format(prefix, tag.replace(':', '='))] = 1.0

    if word in word2vec:
        v = word2vec[word]
        for i, x in enumerate(v):
            if x > 0.0:
                features['0_{}[{}]'.format(i, prefix)] = x
            elif x < 0.0:
                features['1_{}[{}]'.format(i, prefix)] = -x

    return features


def vectorize_sample(words, word2vec, word2tags):
    lines2 = []
    nb_words = len(words)
    for iword, word in enumerate(words):
        word_features = dict()
        for j in range(-winspan, winspan+1):
            iword2 = iword + j
            if nb_words > iword2 >= 0:
                features = get_word_features(words[iword2], str(j), word2vec, word2tags)
                word_features.update(features)

        lines2.append(word_features)

    return lines2


config_path = os.path.join('../../tmp', 'postagger.config')
model_config = json.load(open(config_path, 'r'))

word2vector_path = model_config['w2v_path']
wordchar2vector_path = model_config['wc2v_path']
winspan = model_config['winspan']
use_gren = model_config['use_gren']
model_path = model_config['model_path']

# Загружаем обученную модель CRF
tagger = pycrfsuite.Tagger()
tagger.open(model_path)

# Загружаем грамматический словарь
word2tags = {BEG_TOKEN: [BEG_TOKEN], END_TOKEN: [END_TOKEN]}
if use_gren:
    print(u'Loading grammar dictionary from {}...'.format(gren_path))
    with codecs.open(gren_path, 'r', 'utf-8') as rdr:
        for line in rdr:
            tx = line.strip().split('\t')
            if len(tx) == 4:
                word = normalize_word(tx[0])
                pos = tx[1]
                tags = tx[3].split(' ')
                tags = set(itertools.chain([pos], tags))
                if word not in word2tags:
                    word2tags[word] = tags
                else:
                    word2tags[word].update(tags)


print(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path))
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])

print(u'Loading the w2v model {}'.format(word2vector_path))
w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
w2v_dims = len(w2v.syn0[0])

word_dims = w2v_dims + wc2v_dims

while True:
    sent = utils.console_helpers.input_kbd(':> ')
    words = [BEG_TOKEN] + sent.strip().split() + [END_TOKEN]

    word2vec = dict()
    for word in words:
        v = np.zeros(word_dims)
        if word in wc2v:
            v[w2v_dims:] = wc2v[word]

        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

    X = vectorize_sample(words, word2vec, word2tags)
    y_pred = tagger.tag(X)

    for word, tag in zip(words, y_pred):
        print(u'{:20s} ==> {}'.format(word, tag))

    print('\n')

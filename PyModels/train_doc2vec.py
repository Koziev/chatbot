# -*- coding: utf-8 -*-
"""
Построение модели doc2vec с помощью инструментов gensim
Визуализация получающихся векторов фраз через kmeans
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import codecs
import gensim
import logging

import sklearn.cluster
import scipy.spatial.distance

#from gensim.models.doc2vec import TaggedDocument

from utils.tokenizer import Tokenizer


dim = 32  # размерность векторов предложений

corpora = ['evaluate_relevancy.txt', 'qa.txt', 'premises.txt', 'paraphrases.txt', 'facts4.txt', 'facts5.txt', 'facts6.txt']
data_folder = '../data'
tmp_folder = '../tmp'

class CorporaReader(object):
    def __init__(self, filenames, tokenizer):
        self.filenames = filenames
        self.tokenizer = tokenizer
        self.label2phrase = dict()

    def __iter__(self):
        sent_index = 0
        self.label2phrase = dict()
        for file_index, filename in enumerate(self.filenames):
            for line_index, line in enumerate(codecs.open(filename, 'r', 'utf-8')):
                line = line.strip()
                if len(line) > 0:
                    if not line.startswith(u'A:'):
                        if line.startswith(u'T:'):
                            line = line.replace(u'T:', u'')
                        elif line.startswith(u'Q:'):
                            line = line.replace(u'Q:', u'')

                        sent_index += 1
                        label = u'SENT_{}'.format(sent_index)
                        words = self.tokenizer.tokenize(line)
                        if len(words) == 0:
                            pass
                        else:
                            self.label2phrase[label] = u' '.join(words)
                            yield gensim.models.doc2vec.TaggedDocument(words=words, tags=[label])

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tokenizer = Tokenizer()

sentences = CorporaReader([os.path.join(data_folder, f) for f in corpora], tokenizer)
model = gensim.models.Doc2Vec(sentences, dm=0, alpha=0.1, size=dim, min_alpha=0.025, iter=10)

if False:
    print('build_vocab...')
    model.build_vocab(sentences)

    print('training doc2vec...')
    for epoch in range(2):
        model.train(sentences)
        model.alpha *= 0.98  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

print('storing...')
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model.save(os.path.join(tmp_folder, 'doc2vec.model'))

with codecs.open(os.path.join(tmp_folder, 'doc2vec.txt'), 'w', 'utf-8') as wrt:
    wrt.write('{} {}\n'.format(len(model.docvecs.doctags), dim))
    for label in model.docvecs.doctags.keys():
        phrase = sentences.label2phrase[label]
        v = model.docvecs[label]
        wrt.write(u'{}\t{}\n'.format(phrase, u' '.join('{:<12.5e}'.format(x) for x in v)))

if True:
    # Визуализация получающихся векторов предложений с помощью kmeans

    nb_sents = len(model.docvecs.doctags.keys())
    sents = []
    vectors = np.zeros((nb_sents, dim), dtype='float32')
    for isent, label in enumerate(model.docvecs.doctags.keys()):
        phrase = sentences.label2phrase[label]
        sents.append(phrase)
        vectors[isent, :] = model.docvecs[label]

    nb_clusters = 100
    print('Start k-means for {0} vectors, {1} clusters...'.format(nb_sents, nb_clusters))
    # (codebook,labels) = scipy.cluster.vq.kmeans2( data=sent_vect, k=n_cluster )
    kmeans = sklearn.cluster.KMeans(n_clusters=nb_clusters, max_iter=20, verbose=1, copy_x=True, n_jobs=1,
                                    algorithm='auto')
    kmeans.fit(vectors)
    labels = kmeans.labels_
    codebook = kmeans.cluster_centers_
    print('Finished.')

    res_path = '../tmp/doc2vec_clusters.txt'
    print('Printing clusters to {0}...'.format(res_path))
    sent_vec_list = vectors
    with codecs.open(res_path, 'w', 'utf-8') as wrt:
        for target_cluster in range(nb_clusters):
            print('{0}/{1}'.format(target_cluster, nb_clusters), end='\r')
            sys.stdout.flush()

            cluster_coord = codebook[target_cluster]

            sent_in_cluster = [(sents[isent], sent_vec_list[isent]) for (isent, l) in enumerate(labels) if
                               l == target_cluster]
            # sent_in_cluster = sorted(sent_in_cluster, key=lambda z: -scipy.spatial.distance.cosine(cluster_coord, z[1]))
            sent_in_cluster = sorted(sent_in_cluster,
                                     key=lambda z: -scipy.spatial.distance.euclidean(cluster_coord, z[1]))
            sent_in_cluster = sent_in_cluster[: min(20, len(sent_in_cluster))]
            sent_in_cluster = [(s, scipy.spatial.distance.euclidean(cluster_coord, v)) for (s, v) in sent_in_cluster]

            wrt.write('\n\ncluster #{}\n'.format(target_cluster))
            for sent, cos_sim in sent_in_cluster:
                wrt.write(u'{}\t{}\n'.format(cos_sim, sent))

    print('All done.')

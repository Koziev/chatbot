# -*- coding: utf-8 -*-
'''
Тренировка LSA.
Сохраняем векторы всех фраз в датасетах qa и relevancy.
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import pickle
import codecs
import sys

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import sklearn.cluster
import scipy.spatial.distance

from utils.tokenizer import Tokenizer

input_path1 = '../data/premise_question_answer.csv'
tmp_folder = '../tmp'
data_folder = '../data'

LSA_DIMS = 60

# -------------------------------------------------------------------

df = pd.read_csv(input_path1, encoding='utf-8', delimiter='\t', quoting=3)
print('samples.count={}'.format(df.shape[0]))

print('Buidling tf-idf corpus...')
tfidf_corpus = set()
tokenizer = Tokenizer()
for i,record in df.iterrows():
    for phrase in [record['premise'], record['question']]:
        phrase = u' '.join(tokenizer.tokenize(phrase))
        tfidf_corpus.add(phrase)

tfidf_corpus = list(tfidf_corpus)
print('{} phrases in tfidf corpus'.format(len(tfidf_corpus)))

print('Fitting LSA...')
vectorizer = TfidfVectorizer(max_features=None, ngram_range=(3, 5), min_df=1, analyzer='char')
svd_model = TruncatedSVD(n_components=LSA_DIMS, algorithm='randomized', n_iter=20, random_state=42)
svd_transformer = Pipeline([('tfidf', vectorizer), ('svd', svd_model)])
svd_transformer.fit(tfidf_corpus)

print('Calculating LSA vectors for phrases...')
phrase_ls = svd_transformer.transform(tfidf_corpus)

print('Storing LSA...')
with open(os.path.join(tmp_folder,'lsa_model.pickle'), 'w') as f:
    lsa = { 'phrases': tfidf_corpus,
            'vectors': phrase_ls,
            'lsa_dims': LSA_DIMS}
    pickle.dump(lsa, f)


if True:
    # Визуализация получающихся векторов предложений с помощью kmeans
    nb_sents = len(tfidf_corpus)
    sents = tfidf_corpus
    vectors = phrase_ls

    nb_clusters = 100
    print('Start k-means for {0} vectors, {1} clusters...'.format(nb_sents, nb_clusters))
    # (codebook,labels) = scipy.cluster.vq.kmeans2( data=sent_vect, k=n_cluster )
    kmeans = sklearn.cluster.KMeans(n_clusters=nb_clusters, max_iter=20, verbose=1, copy_x=True, n_jobs=1,
                                    algorithm='auto')
    kmeans.fit(vectors)
    labels = kmeans.labels_
    codebook = kmeans.cluster_centers_
    print('Finished.')

    res_path = '../tmp/lsa_clusters.txt'
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

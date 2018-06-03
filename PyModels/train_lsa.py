# -*- coding: utf-8 -*-
'''
Тренировка LSA модели, оценка на задаче выбора релевантной предпосылки,
кластеризация для визуализации.

Сохраняем векторы всех фраз, присутствующих в датасетах для qa и relevancy.
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import os
import pickle
import codecs
import sys
import itertools
import numpy as np

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import sklearn.cluster
import scipy.spatial.distance

from utils.tokenizer import Tokenizer
from trainers.evaluation_dataset import EvaluationDataset
from trainers.evaluation_markup import EvaluationMarkup


tmp_folder = '../tmp'
data_folder = '../data'

LSA_DIMS = 60


def v_cosine( a, b ):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# -------------------------------------------------------------------

print('Buidling tf-idf corpus...')
tfidf_corpus = set()
tokenizer = Tokenizer()

for fname in ['premise_question_answer.csv', 'premise_question_relevancy.csv']:
    df = pd.read_csv(os.path.join(data_folder, fname),
                     encoding='utf-8',
                     delimiter='\t',
                     quoting=3)
    for i, record in df.iterrows():
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


# -----------------------------------------------------------

# Оценим точность выбора релевантного вопроса для предпосылки по специальному
# датасету.
eval_data = EvaluationDataset(0, tokenizer)
eval_data.load(data_folder)

all_phrases = eval_data.get_all_phrases()
all_vectors = svd_transformer.transform(all_phrases)

phrase2vec = dict(itertools.izip(all_phrases, all_vectors))

nb_good = 0
nb_bad = 0

for irecord, phrases in eval_data.generate_groups():
    y_pred = []
    for irow, (premise_words, question_words) in enumerate(phrases):
        premise = u' '.join(premise_words)
        premise_v = phrase2vec[premise] if premise in phrase2vec else None

        question = u' '.join(question_words)
        question_v = phrase2vec[question] if question in phrase2vec else None

        if premise_v is not None and question_v is not None:
            sim = v_cosine(premise_v, question_v)
        else:
            sim = 0.0
        y_pred.append(sim)

    # предпосылка с максимальной релевантностью
    max_index = np.argmax(y_pred)
    selected_premise = u' '.join(phrases[max_index][0]).strip()

    # эта выбранная предпосылка соответствует одному из вариантов
    # релевантных предпосылок в этой группе?
    if eval_data.is_relevant_premise(irecord, selected_premise):
        nb_good += 1
        print(EvaluationMarkup.ok_color + EvaluationMarkup.ok_bullet + EvaluationMarkup.close_color, end='')
    else:
        nb_bad += 1
        print(EvaluationMarkup.fail_color + EvaluationMarkup.fail_bullet + EvaluationMarkup.close_color, end='')

    max_sim = np.max(y_pred)

    question_words = phrases[0][1]
    print(u'{:<40} {:<40} {}/{}'.format(u' '.join(question_words), u' '.join(phrases[max_index][0]), y_pred[max_index],
                                        y_pred[0]))

# Итоговая точность выбора предпосылки.
accuracy = float(nb_good) / float(nb_good + nb_bad)
print('accuracy={}'.format(accuracy))

# ----------------------------------------------------------------------------

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

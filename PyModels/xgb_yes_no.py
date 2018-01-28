# -*- coding: utf-8 -*-
'''
Тренировка модели классификации yes/no двух фраз (предпосылка и вопрос) для
вопросно-ответной системы https://github.com/Koziev/chatbot.
Используется XGBoost.
Датасет должен быть сгенерирован и находится в папке ../data (см. prepare_qa_dataset.py)
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import codecs
import itertools
import json
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer

# Основной гиперпараметр модели - число символов в N-граммах, мешок которых
# представляет анализируемое предложение.
SHINGLE_LEN = 3

# -------------------------------------------------------------------

input_path = '../data/premise_question_answer.csv'
tmp_folder = '../tmp'
data_folder = '../data'

# -------------------------------------------------------------------

BEG_WORD = '\b'
END_WORD = '\n'

# -------------------------------------------------------------------

def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str( words ):
    return u' '.join(itertools.chain([BEG_WORD], words, [END_WORD]))

# -------------------------------------------------------------------

df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
print('samples.count={}'.format(df.shape[0]))

nb_yes = 0 # кол-во ответов "да"
nb_no = 0 # кол-во ответов "нет"

all_shingles = set()

tokenizer = Tokenizer()
for i,record in df.iterrows():
    for phrase in [record['premise'], record['question']]:
        words = tokenizer.tokenize(phrase)
        wx = words2str(words)
        all_shingles.update( ngrams(wx, SHINGLE_LEN) )

    answer = record['answer'].lower().strip()
    if answer==u'да':
        nb_yes +=1
    elif answer==u'нет':
        nb_no += 1

nb_shingles = len(all_shingles)
print('nb_shingles={}'.format(nb_shingles))

print('nb_yes={}'.format(nb_yes))
print('nb_no={}'.format(nb_no))

shingle2id = dict([(s,i) for i,s in enumerate(all_shingles)])

# --------------------------------------------------------------------------

def vectorize_sample_x( X_data, idata, premise_shingles, question_shingles, shingle2id):
    ps = set(premise_shingles)
    qs = set(question_shingles)
    common_shingles = ps & qs
    notmatched_ps = ps - qs
    notmatched_qs = qs - ps

    nb_shingles = len(shingle2id)

    icol = 0
    for shingle in common_shingles:
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_ps:
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_qs:
        X_data[idata, icol+shingle2id[shingle]] = True

# --------------------------------------------------------------------------

nb_patterns = 0

for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Counting yes/no patterns'):
    premise = row['premise']
    question = row['question']
    answer = row['answer'].lower()

    if answer in [u'да', u'нет']:
        nb_patterns += 1


nb_features = nb_shingles*3
X_data = lil_matrix((nb_patterns, nb_features), dtype='bool')
y_data = []
idata = 0

for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Vectorization'):
    premise = row['premise']
    question = row['question']
    answer = row['answer'].lower()

    if answer in [u'да', u'нет']:
        y = 1 if answer==u'да' else 0
        y_data.append(y)

        premise_words = tokenizer.tokenize(premise)
        question_words = tokenizer.tokenize(question)

        premise_shingles = ngrams( words2str(premise_words), SHINGLE_LEN )
        question_shingles = ngrams( words2str(question_words), SHINGLE_LEN )
        vectorize_sample_x( X_data, idata, premise_shingles, question_shingles, shingle2id )

        idata += 1


nb_0 = len(filter(lambda y:y==0, y_data))
nb_1 = len(filter(lambda y:y==1, y_data))

print('nb_0={}'.format(nb_0))
print('nb_1={}'.format(nb_1))

# ------------------------------------------------------------------------

# Сгенерируем внятные имена фич, чтобы увидеть алгоритм классификации в xgb
feature_names = [u'common('+shingle+u')' for shingle,i in sorted(shingle2id.iteritems(),key=lambda z:z[1]) ]
feature_names.extend( [u'premise('+shingle+u')' for shingle,i in sorted(shingle2id.iteritems(),key=lambda z:z[1]) ] )
feature_names.extend( [u'quest('+shingle+u')' for shingle,i in sorted(shingle2id.iteritems(),key=lambda z:z[1]) ] )


# -----------------------------------------------------------------------


SEED = 123456
TEST_SHARE = 0.2
X_train, X_val, y_train, y_val = train_test_split( X_data, y_data, test_size=TEST_SHARE, random_state=SEED )

D_train = xgboost.DMatrix(X_train, y_train, feature_names=feature_names, silent=0)
D_val = xgboost.DMatrix(X_val, y_val, feature_names=feature_names, silent=0)

xgb_params = {
    'booster': 'gbtree',
    # 'n_estimators': _n_estimators,
    'subsample': 1.0,
    'max_depth': 4,
    'seed': 123456,
    'min_child_weight': 1,
    'eta': 0.10,
    'gamma': 0.01,
    'colsample_bytree': 1.0,
    'colsample_bylevel': 1.0,
    'eval_metric': 'error',
    'objective': 'binary:logistic',
    'silent': 1,
    # 'updater': 'grow_gpu'
}

print('Train model...')
cl = xgboost.train(xgb_params,
                   D_train,
                   evals=[(D_val, 'val')],
                   num_boost_round=1000,
                   verbose_eval=10,
                   early_stopping_rounds=50)

print('Training is finished')
y_pred = cl.predict(D_val)
y_pred = (y_pred >= 0.5).astype(np.int)
score = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred)
print('score={}'.format(score))


model_filename = os.path.join( tmp_folder, 'xgb_yes_no.model' )

# сохраним конфиг модели, чтобы ее использовать в чат-боте
model_config = {
                'shingle2id': shingle2id,
                'model_filename': model_filename,
                'shingle_len': SHINGLE_LEN,
                'nb_features': nb_features,
                'feature_names': feature_names
               }

with open(os.path.join(tmp_folder,'xgb_yes_no.config'), 'w') as f:
    json.dump(model_config, f)


cl.save_model( model_filename )


# Визуализация веса фич
feature_scores = cl.get_fscore()
with codecs.open('../tmp/feature_xgboost_scores.txt', 'w', 'utf-8') as wrt:
    for (feature, score) in sorted(feature_scores.iteritems(), key=lambda z: -z[1]):
        wrt.write(u'{}\t{}\n'.format(feature, score))

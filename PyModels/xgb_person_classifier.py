# -*- coding: utf-8 -*-
"""
Модели для определения грамматического лица у фразы на базе XGB классификатора.
Модель используется в проекте вопросно-ответной системы https://github.com/Koziev/chatbot

Используются ранее сгенерированные датасеты:
change_person_1s_2s_dataset_4.csv
change_person_1s_2s_dataset_5.csv
...
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import codecs
import itertools
import json
import os

import sklearn.metrics
import tqdm
import xgboost
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer

SHINGLE_LEN = 3

manual_datasets = ['../data/names_person_classif.txt', '../data/person_classifications.txt']

input_paths_1s = ['../data/change_person_1s_2s_dataset_4.csv',
                  '../data/change_person_1s_2s_dataset_5.csv']

# отсюда возьмем список вопросов с подлежащем в 3м лице
qa_paths_3 = ['../data/premise_question_answer6.txt',
              '../data/premise_question_answer5.txt',
              '../data/premise_question_answer4.txt',
              '../data/premise_question_answer_neg4.txt',
              '../data/premise_question_answer_neg5.txt',
              '../data/premise_question_answer_names4.txt'
              ]

# отсюда возьмем список вопросов с подлежащем в 1м лице
qa_paths_1s = ['../data/premise_question_answer4_1s.txt',
               '../data/premise_question_answer5_1s.txt',
               '../data/premise_question_answer_neg4_1s.txt',
               '../data/premise_question_answer_neg5_1s.txt',
               '../data/premise_question_answer_names4_1s.txt'
               ]

# отсюда возьмем список вопросов с подлежащем во 2м лице
qa_paths_2s = ['../data/premise_question_answer4_2s.txt',
               '../data/premise_question_answer5_2s.txt',
               '../data/premise_question_answer_neg4_2s.txt',
               '../data/premise_question_answer_neg5_2s.txt',
               '../data/premise_question_answer_names4_2s.txt'
               ]

tmp_folder = '../tmp'
data_folder = '../data'

# -------------------------------------------------------------------

BEG_WORD = '\b'
END_WORD = '\n'

SEED = 123456
TEST_SHARE = 0.2


# -------------------------------------------------------------------

def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    return u' '.join(itertools.chain([BEG_WORD], words, [END_WORD]))

# -------------------------------------------------------------------

config_path = os.path.join(tmp_folder, 'xgb_person_classifier.config')


tokenizer = Tokenizer()

# для тренировки классификатора нам надо взять все предпосылки из датасетов
# по группам 1е, 2е, 3е лицо.
input_phrases = []
classif_ys = []

for ds_path in manual_datasets:
    print(u'Processing {}\t\t'.format(ds_path), end='')
    with codecs.open(ds_path, 'r', 'utf-8') as rdr:
        n = 0
        for iline, line in enumerate(rdr):
            line = line.strip()
            if len(line) > 0:
                parts = line.split('|')
                if len(parts) != 2:
                    print(u'Error in dataset \"{}\", in line #{}=\"{}\": two fields separated by | expected'.format(ds_path, iline, line.strip()))
                    raise RuntimeError()
                premise = parts[0]
                input_phrases.append(premise)
                classif_ys.append(str(parts[1]))
                n +=1
        print('{} samples'.format(n))

for (ds_paths, ds_person) in [(qa_paths_1s, '1s'), (qa_paths_2s, '2s'), (qa_paths_3, '3')]:
    for ds_path in ds_paths:
        print(u'Processing {}\t\t'.format(ds_path), end='')
        with codecs.open(ds_path, 'r', 'utf-8') as rdr:
            n = 0
            for line in rdr:
                if line.startswith(u'T:'):
                    premise = line.replace(u'T:', u'').strip()
                    input_phrases.append(premise)
                    classif_ys.append(ds_person)
                elif line.startswith(u'Q:'):
                    premise = line.replace(u'Q:', u'').strip()
                    input_phrases.append(premise)
                    classif_ys.append(ds_person)
                n += 1
            print('{} samples'.format(n))

# также добавим фраз из датасетов для тренировки менятеля лица
for ds_path in input_paths_1s:
    print(u'Processing {}\t\t'.format(ds_path), end='')
    with codecs.open(ds_path, 'r', 'utf-8') as rdr:
        n = 0
        for line in rdr:
            cols = line.strip().split(u'\t')
            phrase_1s = cols[0]
            phrase_2s = cols[1]
            input_phrases.append(phrase_1s)
            classif_ys.append('1s')
            input_phrases.append(phrase_2s)
            classif_ys.append('2s')
            n += 1
        print('{} samples'.format(n))

print('Total number of patterns={}'.format(len(input_phrases)))

n_1s = len(filter(lambda z:z == '1s', classif_ys))
n_2s = len(filter(lambda z:z == '2s', classif_ys))
n_3 = len(filter(lambda z:z == '3', classif_ys))
print('Classes:\n1s => {}\n2s => {}\n3  => {}'.format(n_1s, n_2s, n_3))


all_shingles = set()
tokenizer = Tokenizer()
for phrase in input_phrases:
    words = tokenizer.tokenize(phrase)
    wx = words2str(words)
    all_shingles.update(ngrams(wx, SHINGLE_LEN))

nb_shingles = len(all_shingles)
print('nb_shingles={}'.format(nb_shingles))

shingle2id = dict((s,i) for i,s in enumerate(all_shingles))

nb_features = nb_shingles

# ------------------------------------------------------------------

nb_patterns = len(input_phrases)
print('Vectorization of {} phrases'.format(nb_patterns))

X_data = lil_matrix((nb_patterns, nb_features), dtype='bool')
y_data = []

for idata,(phrase,y) in tqdm.tqdm(enumerate(itertools.izip(input_phrases, classif_ys)), total=nb_patterns):
    words = tokenizer.tokenize(phrase)
    wx = words2str(words)
    shingles = ngrams(wx, SHINGLE_LEN)
    for shingle in shingles:
        X_data[idata, shingle2id[shingle]] = True
    if y == '1s':
        y_data.append(0)
    elif y == '2s':
        y_data.append(1)
    else:
        y_data.append(2)

# ------------------------------------------------------------------

SEED = 123456
TEST_SHARE = 0.2
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=TEST_SHARE, random_state=SEED)

D_train = xgboost.DMatrix(X_train, y_train, silent=0)
D_val = xgboost.DMatrix(X_val, y_val, silent=0)

xgb_params = {
    'booster': 'gbtree',
    # 'n_estimators': _n_estimators,
    'subsample': 1.0,
    'max_depth': 4,
    'seed': 123456,
    'min_child_weight': 1,
    'eta': 0.30,
    'gamma': 0.01,
    'colsample_bytree': 1.0,
    'colsample_bylevel': 1.0,
    'eval_metric': 'merror',
    'objective': 'multi:softmax',
    'num_class': 3,
    'silent': 1,
    # 'updater': 'grow_gpu'
}

print('Train model...')
cl = xgboost.train(xgb_params,
                   D_train,
                   evals=[(D_val, 'val')],
                   num_boost_round=5000,
                   verbose_eval=50,
                   early_stopping_rounds=50)

print('Training is finished')
y_pred = cl.predict(D_val)
#y_pred = [int(y) for y in y_pred]
score = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_pred)
print('score={}'.format(score))

model_filename = os.path.join( tmp_folder, 'xgb_person_classifier.model' )
cl.save_model( model_filename )

# ---------------------------------------------

# сохраним конфиг модели, чтобы ее использовать в чат-боте
model_config = {
    'engine': 'xgb',
    'model_folder': tmp_folder,
    'shingle_len': SHINGLE_LEN,
    'shingle2id': shingle2id,
    'nb_features': nb_features,
    'model_filename': model_filename
}

with open(config_path, 'w') as f:
    json.dump(model_config, f)

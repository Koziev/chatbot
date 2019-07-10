# coding: utf-8
"""
Тренер классификатора интентов для чатбота.
"""

from __future__ import print_function
import pickle
import pandas as pd
import json
import os
import logging
import logging.handlers
import numpy as np
import argparse
import platform

from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier

from utils.console_helpers import input_kbd


BEG_CHAR = u'\b'
END_CHAR = u'\n'


def get_params_str(model_params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in model_params.items())


def data_vectorization(samples, model_params):
    labels = samples['intent'].values
    label2index = dict((label, i) for (i, label) in enumerate(set(labels)))

    y_data = np.asarray([label2index[label] for label in labels])

    phrases = samples['phrase'].values

    vectorizer = TfidfVectorizer(analyzer='char',
                                 ngram_range=(model_params['min_shingle_len'],
                                              model_params['max_shingle_len']),
                                 min_df=model_params['min_df'],
                                 use_idf=False,
                                 norm=None)
    X_data = vectorizer.fit_transform(phrases)

    # 09.07.2019 в lookup сохраним короткие сэмплы и сэмплы для интентов с <= 10 сэмплами
    label2samples = dict()
    for label, sample in zip(labels, phrases):
        if label not in label2samples:
            label2samples[label] = []
        label2samples[label].append(sample.lower())

    phrase2intent = dict()

    # Все фразы в классах с 10 сэмплами и менее
    for label in [label for label, samples in label2samples.items() if len(samples) <= 10]:
        for phrase in label2samples[label]:
            if phrase not in phrase2intent:
                phrase2intent[phrase] = label
            else:
                logging.warning(u'Ambigous labeling of phrase "%s" as "%s" and "%s" intents', phrase, label,
                                phrase2intent[phrase])

    # Короткие сэмплы
    for label, samples in label2samples.items():
        for phrase in samples:
            if len(phrase) < 10 or phrase.count(u' ') < 2:
                if phrase not in phrase2intent:
                    phrase2intent[phrase] = label
                else:
                    if label != phrase2intent[phrase]:
                        logging.warning(u'Ambiguous labeling of phrase "%s" as "%s" and "%s" intents', phrase, label,
                                        phrase2intent[phrase])

    logging.info('%d samples in lookup table', len(phrase2intent))

    # Для выполнения быстрого нечеткого сопоставления построим обратный индекс по шинглам
    shingle2phrases = dict()
    for phrase in phrase2intent.keys():
        padded_phrase = u'[' + phrase + u']'
        for shingle in set((z1+z2+z3) for z1, z2, z3 in zip(padded_phrase, padded_phrase[1:], padded_phrase[2:])):
            if shingle not in shingle2phrases:
                shingle2phrases[shingle] = []
            shingle2phrases[shingle].append(phrase)

    return vectorizer, X_data, y_data, label2index, phrase2intent, shingle2phrases


def create_estimator(model_params):
    if model_params['engine'] == 'SGDClassifier':
        cl = SGDClassifier(loss='hinge',
                           penalty=model_params['penalty'], alpha=0.0001, l1_ratio=0.15,
                           max_iter=10, tol=None, shuffle=True,
                           verbose=1, epsilon=0.1, n_jobs=4, random_state=None,
                           learning_rate='optimal', eta0=0.0, power_t=0.5,
                           # early_stopping=True, validation_fraction=0.1,
                           average=False, n_iter=None)
        return cl
    elif model_params['engine'] == 'LogisticRegression':
        cl = LogisticRegression(penalty=model_params['penalty'],
                                tol=0.0001,
                                C=model_params['C'],
                                n_jobs=4,
                                multi_class='multinomial', verbose=0, solver='lbfgs',
                                max_iter=1000)
        return cl
    elif model_params['engine'] == 'LinearSVC':
        cl = LinearSVC(penalty=model_params['penalty'],
                       loss='squared_hinge',
                       dual=False,
                       tol=0.0001,
                       C=model_params['C'],
                       multi_class='ovr',
                       fit_intercept=True,
                       intercept_scaling=1,
                       class_weight=None,
                       verbose=0,
                       random_state=None,
                       max_iter=2000)
        return cl
    elif model_params['engine'] == 'SVC':
        cl = SVC(#penalty=model_params['penalty'],
                 #loss='squared_hinge',
                 #dual=False,
                 tol=0.0001,
                 C=model_params['C'],
                 #multi_class='ovr',
                 #fit_intercept=True,
                 #intercept_scaling=1,
                 class_weight=None,
                 verbose=0,
                 random_state=None,
                 max_iter=2000)
        return cl
    elif model_params['engine'] == 'GBM':
        cl = GradientBoostingClassifier(loss='deviance',
                                        learning_rate=model_params['learning_rate'],
                                        n_estimators=model_params['n_estimators'],
                                        subsample=model_params['subsample'],
                                        criterion='friedman_mse',
                                        min_samples_split=model_params['min_samples_split'],
                                        min_samples_leaf=model_params['min_samples_leaf'],
                                        min_weight_fraction_leaf=0.0,
                                        max_depth=model_params['max_depth'],
                                        #min_impurity_split=1e-07,
                                        init=None,
                                        random_state=1234567,
                                        #max_features=None,
                                        verbose=0,
                                        max_leaf_nodes=None, warm_start=False, presort='auto')
        return cl
    else:
        raise NotImplementedError('Model engine={} is not implemented'.format(model_params['engine']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier trainer for intent detection')
    parser.add_argument('--run_mode', choices='gridsearch train query'.split(), default='gridsearch')
    parser.add_argument('--tmp_dir', default='../tmp')
    parser.add_argument('--dataset', default='../data/intents_dataset.csv')
    args = parser.parse_args()

    tmp_dir = args.tmp_dir
    dataset_path = args.dataset
    run_mode = args.run_mode

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    lf = logging.FileHandler(os.path.join(tmp_dir, 'train_intent_classifier.log'), mode='w')
    lf.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)

    logging.info('Start using dataset_path={}'.format(dataset_path))

    # файл в формате json с найденными оптимальными параметрами классификатора,
    # создается в ходе gridsearch, используется для train
    best_params_path = os.path.join(tmp_dir, 'train_intent_classifier.best_params.json')

    # файл с обученной моделью, создается в train, используется в query
    model_path = os.path.join(tmp_dir, 'intent_classifier.model')

    df = pd.read_csv(dataset_path,
                     encoding='utf-8',
                     delimiter='\t',
                     index_col=None,
                     keep_default_na=False)

    if run_mode == 'gridsearch':
        logging.info('=== GRIDSEARCH ===')

        best_model_params = None
        best_acc = 0.0

        model_params = dict()

        for min_shingle_len in [2, 3]:
            model_params['min_shingle_len'] = min_shingle_len
            for max_shingle_len in [3, 4]:
                model_params['max_shingle_len'] = max_shingle_len
                for min_df in [1, 2]:
                    model_params['min_df'] = min_df

                    vectorizer, X_data, y_data, label2index, phrase2label, shingle2phrases = data_vectorization(df, model_params)

                    for engine in ['LinearSVC', 'SVC']:  #, 'GBM', 'LogisticRegression'
                        model_params['engine'] = engine

                        if engine == 'GBM':
                            for learning_rate in [0.05, 0.1, 0.3]:
                                model_params['learning_rate'] = learning_rate

                                for n_estimators in [50, 100, 200]:
                                    model_params['n_estimators'] = n_estimators

                                    for subsample in [1.0, 0.8]:
                                        model_params['subsample'] = subsample

                                        for min_samples_split in [2]:
                                            model_params['min_samples_split'] = min_samples_split

                                            for min_samples_leaf in [1]:
                                                model_params['min_samples_leaf'] = min_samples_leaf

                                                for max_depth in [3, 4, 5]:
                                                    model_params['max_depth'] = max_depth

                                                    logging.info(u'Cross-validation using model_params: {}'.format(
                                                        get_params_str(model_params)))

                                                    estimator = create_estimator(model_params)
                                                    cv_res = cross_val_score(estimator, X_data, y_data,
                                                                             scoring='accuracy', cv=7, n_jobs=8,
                                                                             verbose=1)
                                                    acc = np.mean(cv_res)

                                                    logging.info(u'k-fold validation accuracy={}'.format(acc))

                                                    if acc > best_acc:
                                                        logging.info('!!! NEW BEST !!! acc={} params={}'.format(acc, get_params_str(model_params)))
                                                        best_acc = acc
                                                        best_model_params = model_params
                                                        with open(best_params_path, 'w') as f:
                                                            json.dump(best_model_params, f, indent=4)
                                                    else:
                                                        logging.info('No improvement over current best_acc={}'.format(best_acc))
                        else:
                            for penalty in ['l2']:
                                model_params['penalty'] = penalty
                                for model_C in [1.0, 100.0, 1000.0, 10000.0]:
                                    model_params['C'] = model_C

                                    logging.info(u'Cross-validation using model_params: {}'.format(get_params_str(model_params)))

                                    estimator = create_estimator(model_params)
                                    cv_res = cross_val_score(estimator, X_data, y_data,
                                                             scoring='accuracy', cv=7, n_jobs=8,
                                                             verbose=1)
                                    acc = np.mean(cv_res)

                                    logging.info(u'k-fold validation accuracy={}'.format(acc))

                                    if acc > best_acc:
                                        logging.info('!!! NEW BEST !!! acc={} params={}'.format(acc, get_params_str(model_params)))
                                        best_acc = acc
                                        best_model_params = model_params
                                        with open(best_params_path, 'w') as f:
                                            json.dump(best_model_params, f, indent=4)
                                    else:
                                        logging.info('No improvement over current best_acc={}'.format(best_acc))

        logging.info('best_acc={} for model_params: {}'.format(best_acc, get_params_str(best_model_params)))

    if run_mode == 'train':
        logging.info('Loading best_model_params from "%s"', best_params_path)
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        best_vectorizer, X_data, y_data, label2index, phrase2label, shingle2phrases = data_vectorization(df, best_model_params)

        # финальное обучение классификатора на всех данных
        logging.info('Training the final classifier model_params: {}'.format(get_params_str(best_model_params)))
        estimator = create_estimator(best_model_params)
        estimator.fit(X_data, y_data)

        # сохраним натренированный классификатор и дополнительные параметры, необходимые для
        # использования модели в чатботе.
        index2label = dict((index, label) for (label, index) in label2index.items())
        model = {'index2label': index2label,
                 'vectorizer': best_vectorizer,
                 'estimator': estimator,
                 'phrase2label': phrase2label,
                 'shingle2phrases': shingle2phrases}
        logging.info(u'Storing model to "%s"', model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info('All done.')

    if run_mode == 'query':
        logging.info('Restoring model from "%s"', model_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        while True:
            phrase = input_kbd(':> ').strip()

            X_query = model['vectorizer'].transform([phrase])
            y_query = model['estimator'].predict(X_query)
            intent_index = y_query[0]
            intent_name = model['index2label'][intent_index]
            print(u'{}'.format(intent_name))


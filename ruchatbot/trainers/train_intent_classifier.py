# coding: utf-8
"""
Тренер классификатора интентов для чатбота.
Реализован подбор параметров в gridsearch, тренировка модели для использовании
в чатботе с сохранением на диск и интерактивная проверка обученной модели.

29.07.2019 - добавлен вывод confusion matrix
29.07.2019 - добавлен перебор параметра norm в TfIdfVectorizer, так
как для несбалансированных классов способ нормализации или отсутствие ее
может сильно влиять на качество линейных моделей.
22.08.2019 - добавлена выдача отчета по confusion matrix
25.08.2019 - добавлена визуализация t-SNE
18.06.2020 - ряд классификаций выделен в отдельные модели (abusiveness, sentiment, direction)
"""

from __future__ import print_function
import pickle
import pandas as pd
import json
import os
import io
import logging
import logging.handlers
import numpy as np
import argparse
import collections

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
import terminaltables

from ruchatbot.utils.console_helpers import input_kbd
from ruchatbot.utils.logging_helpers import init_trainer_logging


NFOLD = 4


def get_params_str(model_params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in model_params.items())


def data_vectorization(samples, model_params):
    labels = samples['label'].values
    label2index = dict((label, i) for (i, label) in enumerate(set(labels)))

    y_data = np.asarray([label2index[label] for label in labels], dtype=np.int)

    phrases = samples['phrase'].values

    vectorizer = TfidfVectorizer(analyzer='char',
                                 ngram_range=(model_params['min_shingle_len'],
                                              model_params['max_shingle_len']),
                                 min_df=model_params['min_df'],
                                 use_idf=False,
                                 norm=model_params['norm'])

    if model_params['nlp_transform'] == 'lower':
        phrases = [f.lower() for f in phrases]
    elif model_params['nlp_transform'] == '':
        pass
    else:
        raise NotImplementedError()

    # TODO: вообще методически более правильно было бы делать fit на тренировочном наборе, а уже transform
    # делать на всех данных.
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
                logging.warning(u'Ambiguous labeling of phrase "%s" as "%s" and "%s" intents', phrase, label,
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


def create_estimator(model_params, computed_params):
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
                                max_iter=2000)
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
                       max_iter=5000)
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
                 gamma='auto',
                 max_iter=5000)
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
    elif model_params['engine'] == 'LightGBM':
        cl = lightgbm.LGBMClassifier(boosting_type='gbdt',
                                     num_leaves=model_params['num_leaves'],  #31,
                                     max_depth=-1,
                                     learning_rate=model_params['learning_rate'],
                                     n_estimators=model_params['n_estimators'],
                                     subsample_for_bin=200000,
                                     #objective='multiclass',
                                     #metric='multi_logloss',
                                     #num_classes=computed_params['nb_labels'],
                                     class_weight=None,
                                     min_split_gain=0.0,
                                     min_child_weight=0.001,
                                     min_child_samples=model_params['min_child_samples'],  #20,
                                     subsample=1.0,
                                     subsample_freq=0,
                                     colsample_bytree=1.0,
                                     reg_alpha=model_params['reg_alpha'],  #0.0,
                                     reg_lambda=model_params['reg_lambda'],  #0.0,
                                     random_state=None,
                                     n_jobs=-1,
                                     silent=True,
                                     importance_type='split')
        return cl
    else:
        raise NotImplementedError('Model engine={} is not implemented'.format(model_params['engine']))


def gridsearch_estimator_params():
    for engine in ['LinearSVC']:  # 'SVC', 'LightGBM', 'GBM', 'LogisticRegression'
        model_params = dict()
        model_params['engine'] = engine

        if engine == 'LightGBM':
            for num_leaves in [20, 31, 40]:
                model_params['num_leaves'] = num_leaves
                for learning_rate in [0.05, 0.1, 0.3]:
                    model_params['learning_rate'] = learning_rate
                    for n_estimators in [100, 200, 300]:
                        model_params['n_estimators'] = n_estimators
                        for min_child_samples in [2, 10, 20]:
                            model_params['min_child_samples'] = min_child_samples
                            for reg_alpha in [0.0]:
                                model_params['reg_alpha'] = reg_alpha
                                for reg_lambda in [0.0]:
                                    model_params['reg_lambda'] = reg_lambda
                                    yield model_params
        elif engine == 'GBM':
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
                                    yield model_params
        else:
            for penalty in ['l2']:
                model_params['penalty'] = penalty
                for model_C in [0.1, 1.0, 100.0, 1000.0, 10000.0]:
                    model_params['C'] = model_C
                    yield model_params


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #plt.show()
    return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier trainer for intent detection')
    parser.add_argument('--run_mode', choices='gridsearch train query'.split(), default='train')
    parser.add_argument('--tmp_dir', default='../tmp')
    parser.add_argument('--datasets', default='../tmp')
    args = parser.parse_args()

    tmp_dir = args.tmp_dir
    dataset_path = os.path.join(args.datasets, 'intents_dataset.csv')
    run_mode = args.run_mode

    init_trainer_logging(os.path.join(tmp_dir, 'train_intent_classifier.log'), logging.DEBUG)

    logging.info('Start "%s" using dataset_path="%s"', run_mode, dataset_path)

    # файл в формате json с найденными оптимальными параметрами классификатора,
    # создается в ходе gridsearch, используется для train
    best_params_path = os.path.join(tmp_dir, 'train_intent_classifier.best_params.json')

    model_names = ['intent', 'abusive', 'sentiment', 'direction']

    if run_mode == 'gridsearch':
        logging.info('=== GRIDSEARCH ===')

        df = pd.read_csv(dataset_path,
                         encoding='utf-8',
                         delimiter='\t',
                         index_col=None,
                         keep_default_na=False)

        best_model_params = None
        best_acc = 0.0

        data_params = dict()

        for min_shingle_len in [2, 3]:
            data_params['min_shingle_len'] = min_shingle_len
            for max_shingle_len in [3, 4, 5, 6]:
                data_params['max_shingle_len'] = max_shingle_len
                for norm in [None, 'l1', 'l2']:
                    data_params['norm'] = norm
                    for min_df in [1, 2]:
                        data_params['min_df'] = min_df
                        for nlp_transform in ['', 'lower']:
                            data_params['nlp_transform'] = nlp_transform

                            vectorizer, X_data, y_data, label2index, phrase2label, shingle2phrases = data_vectorization(df,
                                                                                                                        data_params)
                            computed_params = {'nb_labels': len(label2index)}

                            for estimator_params in gridsearch_estimator_params():

                                model_params = dict()
                                model_params.update(data_params)
                                model_params.update(estimator_params)

                                logging.info(u'Cross-validation using model_params: %s', get_params_str(model_params))

                                estimator = create_estimator(model_params, computed_params)
                                cv_res = cross_val_score(estimator, X_data, y_data,
                                                         scoring='f1_weighted', cv=NFOLD, n_jobs=NFOLD,
                                                         verbose=1)
                                acc = np.mean(cv_res)

                                logging.info(u'k-fold validation accuracy=%f', acc)

                                if acc > best_acc:
                                    logging.info('!!! NEW BEST !!! acc=%f params=%s', acc, get_params_str(model_params))
                                    best_acc = acc
                                    best_model_params = model_params
                                    with open(best_params_path, 'w') as f:
                                        json.dump(best_model_params, f, indent=4)
                                else:
                                    logging.info('No improvement over current best_acc=%f', best_acc)

        logging.info('best_acc=%f for model_params: %s', best_acc, get_params_str(best_model_params))

    if run_mode == 'train':
        logging.info('Loading best_model_params from "%s"', best_params_path)
        with open(best_params_path, 'r') as f:
            best_model_params = json.load(f)

        # 18.06.2020 дополнительные классификаторы тренируем как самостоятельные модели
        for model_name in model_names:
            dataset_path = os.path.join(args.datasets, '{}_dataset.csv'.format(model_name))
            model_path = os.path.join(tmp_dir, '{}_classifier.model'.format(model_name))
            logging.info('Model "%s", dataset="%s", model_path="%s"', model_name, dataset_path, model_path)

            df = pd.read_csv(dataset_path,
                             encoding='utf-8',
                             delimiter='\t',
                             index_col=None,
                             keep_default_na=False)

            best_vectorizer, X_data, y_data, label2index, phrase2label, shingle2phrases = data_vectorization(df, best_model_params)
            computed_params = {'nb_labels': len(label2index)}

            # финальное обучение классификатора на всех данных
            logging.info('Training the final classifier model_params: %s', get_params_str(best_model_params))
            estimator = create_estimator(best_model_params, computed_params)
            estimator.fit(X_data, y_data)

            # сохраним натренированный классификатор и дополнительные параметры, необходимые для
            # использования модели в чатботе.
            index2label = dict((index, label) for (label, index) in label2index.items())
            model = {'index2label': index2label,
                     'vectorizer': best_vectorizer,
                     'estimator': estimator,
                     'nlp_transform': best_model_params['nlp_transform'],
                     'phrase2label': phrase2label,
                     'shingle2phrases': shingle2phrases}

            logging.info(u'Storing model "%s" to "%s"', model_name, model_path)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

            if model_name == 'intent':
                # -------------- Визуализация с помощью SVD + tSNE -----------------------

                # Оставим 10 самых крупных классов.
                label2freq = collections.Counter()
                label2freq.update(y_data)
                top_labels = [index2label[y] for y, _ in label2freq.most_common(10)]
                df2 = df[df.label.isin(top_labels)]
                vectorizer, X_data2, y_data2, _, _, _ = data_vectorization(df2, best_model_params)

                svd_model = TruncatedSVD(n_components=50, algorithm='randomized', n_iter=20, random_state=31415926)
                x2 = svd_model.fit_transform(X_data2)
                tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                tsne_results = tsne.fit_transform(x2)

                df3 = pd.DataFrame(columns='label tsne1 tsne2'.split())
                df3['label'] = df2.label
                df3['tsne1'] = tsne_results[:, 0]
                df3['tsne2'] = tsne_results[:, 1]
                fig = plt.figure(figsize=(10, 10))
                sns.scatterplot(
                    x="tsne1", y="tsne2",
                    hue="label",
                    palette=sns.color_palette("hls", len(set(y_data2))),
                    data=df3,
                    #legend="full",
                    alpha=0.3
                )
                #fig.set_figheight(100)
                #fig.set_figwidth(100)
                fig.savefig(fname=os.path.join(tmp_dir, 'train_intent_classifier.tsne.png'))

                # --------------------- конец tsne ----------------------------


                # Попробуем оценить confusion matrix
                # Для это обучим классификатор еще раз на урезанном датасете, и рассчитаем
                # ошибки по тестовому набору
                X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=True)
                estimator = create_estimator(best_model_params, computed_params)
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                test_acc = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)

                class_names = [l for (l, i) in sorted(label2index.items(), key=lambda z: z[1])]

                confmat = sklearn.metrics.confusion_matrix(y_true=[index2label[y] for y in y_test],
                                                           y_pred=[index2label[y] for y in y_pred],
                                                           labels=class_names)

                with io.open(os.path.join(tmp_dir, 'train_intent_classifier.report.txt'), 'w', encoding='utf-8') as wrt:
                    rep = sklearn.metrics.classification_report(y_true=[index2label[y] for y in y_test],
                                                                y_pred=[index2label[y] for y in y_pred],
                                                                output_dict=True)

                    intent_ranking = []
                    for label, info in rep.items():
                        if label in class_names:
                            f1 = info['f1-score']
                            intent_ranking.append((label, f1))
                    intent_ranking = sorted(intent_ranking, key=lambda z: z[1])
                    for class_name, class_f1 in intent_ranking:
                        wrt.write(u"{:<30s} f1={:.2f}\n".format(class_name, class_f1))

                    wrt.write(u'\n\n')
                    wrt.write(sklearn.metrics.classification_report(y_true=[index2label[y] for y in y_test],
                                                                    y_pred=[index2label[y] for y in y_pred]))

                fig = plot_confusion_matrix(y_test, y_pred, classes=class_names,
                                            title='Confusion matrix of intent classification',
                                            normalize=True)
                fig = fig.get_figure()
                fig.set_figheight(100)
                fig.set_figwidth(100)
                fig.savefig(fname=os.path.join(tmp_dir, 'train_intent_classifier.confusion_matrix.png'))

                logging.info('test_acc=%f', test_acc)
                logging.info('All done.')

    if run_mode == 'query':
        models = []

        for model_name in model_names:
            model_path = os.path.join(tmp_dir, '{}_classifier.model'.format(model_name))
            logging.info('Restoring model from "%s"', model_path)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                models.append((model_name, model))

        while True:
            phrase = input_kbd(':> ').strip()

            table = ['model class'.split()]

            for model_name, model in models:
                if model['nlp_transform'] == 'lower':
                    phrase = phrase.lower()

                X_query = model['vectorizer'].transform([phrase])
                y_query = model['estimator'].predict(X_query)
                intent_index = y_query[0]
                intent_name = model['index2label'][intent_index]
                table.append((model_name, intent_name))

            table = terminaltables.AsciiTable(table)
            print(table.table)



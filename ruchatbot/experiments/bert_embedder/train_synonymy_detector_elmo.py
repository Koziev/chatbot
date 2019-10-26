# coding: utf-8
"""
Тренер детектора синонимичности двух фраз с использованием эмбеддингов ELMO.
Бинарная классификация.
Датасет data/synonymy_dataset.csv должен с парами релевантных/нерелевантных
фраз должен быть заранее подготовлен.
19.10.2019 первая реализация
25-10-2019 добавлен сценарий eval для оценки по полному датасету через кросс-валидацию
"""

from __future__ import print_function
import numpy as np
import argparse
import random
import io
import pandas as pd
import os
import json

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics

import keras.callbacks
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import recurrent
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.layers.merge import concatenate, add, multiply
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
import keras.regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from elmo_keras_layer import ELMoEmbedding


NFOLD = 4
max_seq_len = 20
weights_path = None


def get_params_str(model_params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in model_params.items())


class Sample:
    def __init__(self, phrase1, words1, phrase2, words2, label):
        self.phrase1 = phrase1
        self.words1 = words1
        self.phrase2 = phrase2
        self.words2 = words2
        self.label = label


def load_data(dataset_path, max_samples):
    computed_params = dict()
    samples1 = []
    samples2 = []
    ys = []
    all_words = set()
    max_len = 0

    df = pd.read_csv(dataset_path, encoding='utf-8', delimiter='\t', quoting=3)
    for i, r in df.iterrows():
        label = r['relevance']

        phrase1 = r['premise']
        phrase2 = r['question']

        words1 = phrase1.split()
        words2 = phrase2.split()

        all_words.update(words1)
        all_words.update(words2)

        max_len = max(max_len, len(words1), len(words2))

        samples1.append((phrase1, words1))
        samples2.append((phrase2, words2))
        ys.append(label)

    computed_params['max_len'] = max_len

    word2index = {'': 0}
    for i, w in enumerate(all_words):
        word2index[w] = i+1
    index2word = dict((i, w) for (w, i) in word2index.items())
    computed_params['word2index'] = word2index
    computed_params['index2word'] = index2word
    computed_params['elmo_dim'] = 1024

    samples = []
    for (phrase1, words1), (phrase2, words2), y in zip(samples1, samples2, ys):
        words1 = [word2index[w] for w in words1]
        words1 = sequence.pad_sequences([words1], maxlen=max_len, padding='post', truncating='post')[0]

        words2 = [word2index[w] for w in words2]
        words2 = sequence.pad_sequences([words2], maxlen=max_len, padding='post', truncating='post')[0]

        samples.append(Sample(phrase1, words1, phrase2,  words2, y))

    if len(samples) > max_samples:
        samples = sorted(samples, key=lambda _: random.random())[:max_samples]

    return samples, computed_params


def create_model(model_params, computed_params):
    max_len = computed_params['max_len']
    elmo_dim = computed_params['elmo_dim']
    index2word = computed_params['index2word']

    input1 = Input(shape=(max_len,), dtype=tf.int64, name='input1')
    input2 = Input(shape=(max_len,), dtype=tf.int64, name='input2')

    encoder1 = ELMoEmbedding(idx2word=index2word, output_mode="default", trainable=False)(input1)
    encoder2 = ELMoEmbedding(idx2word=index2word, output_mode="default", trainable=False)(input2)

    addition = add([encoder1, encoder2])
    minus_y1 = Lambda(lambda x: -x, output_shape=(elmo_dim,))(encoder1)
    mul = add([encoder2, minus_y1])
    mul = multiply([mul, mul])
    encoder = keras.layers.concatenate(inputs=[mul, addition, encoder1, encoder2])

    net = encoder
    net = Dense(units=model_params['units1'], activation=model_params['activ1'])(net)
    if model_params['units2'] > 0:
        net = Dense(units=model_params['units2'], activation=model_params['activ2'])(net)

    net = Dense(units=1, name='output', activation='sigmoid')(net)

    model = Model(inputs=[input1, input2], outputs=net)
    model.compile(loss='binary_crossentropy', optimizer=model_params['optimizer'], metrics=['accuracy'])

    return model


def generate_rows(params, computed_params, samples, batch_size, mode):
    batch_index = 0
    batch_count = 0
    max_len = computed_params['max_len']

    X1_batch = np.zeros((batch_size, max_len), dtype=np.int64)
    X2_batch = np.zeros((batch_size, max_len), dtype=np.int64)
    y_batch = np.zeros((batch_size), dtype=np.bool)

    while True:
        used_samples = []
        for irow, sample in enumerate(sorted(samples, key=lambda _: random.random())):
            used_samples.append(sample)
            X1_batch[batch_index, :] = sample.words1
            X2_batch[batch_index, :] = sample.words2
            y_batch[batch_index] = sample.label
            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                x_inputs = {'input1': X1_batch, 'input2': X2_batch}
                if mode == 0:
                    yield x_inputs
                elif mode == 1:
                    yield (x_inputs, {'output': y_batch})
                else:
                    yield (x_inputs, {'output': y_batch}, used_samples)

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


class VisualizeCallback(keras.callbacks.Callback):
    """Так как керасовский метод сохранения и загрузки весов модели не работает с графом, в котором
    участвует BERT, то делаем свой колбэк для записи чекпоинтов через tf.train.Saver.
    """
    def __init__(self, model, model_params, computed_params, val_samples, weights_path):
        self.epoch = 0
        self.model = model
        self.model_params = model_params
        self.computed_params = computed_params
        self.val_samples = val_samples
        self.best_val_acc = -np.inf  # для сохранения самой точной модели
        self.weights_path = weights_path
        self.wait = 0  # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 10
        self.saver = tf.train.Saver()

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1

        print('Epoch {} validation:'.format(self.epoch))
        val_acc = score_model(self.model, self.model_params, self.computed_params, self.val_samples)

        if val_acc > self.best_val_acc:
            print('\nf1 score improved from {} to {}, saving model to {}\n'.format(self.best_val_acc, val_acc, self.weights_path))
            self.best_val_acc = val_acc
            #self.model.save_weights(self.weights_path)
            save_path = self.saver.save(sess, self.weights_path)
            self.wait = 0
        else:
            print('\nf1 score={} did not improve (current best score={})\n'.format(val_acc, self.best_val_acc))
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_best_accuracy(self):
        return self.best_val_acc

    def new_epochs(self):
        self.wait = 0
        self.model.stop_training = False


def train_model(model, params, computed_params, train_samples, val_samples, use_early_stopping=True, verbose=0):
    nb_train_patterns = len(train_samples)
    nb_valid_patterns = len(val_samples)
    print('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_acc'
    callbacks = []

    if False:
        model_checkpoint = ModelCheckpoint(weights_path,
                                           monitor=monitor_metric,
                                           verbose=1,
                                           save_best_only=True,
                                           mode='auto')
        callbacks.append(model_checkpoint)

    if use_early_stopping:
        #early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')
        early_stopping = VisualizeCallback(model, params, computed_params, val_samples, weights_path)
        callbacks.append(early_stopping)

    batch_size = params['batch_size']
    hist = model.fit_generator(generator=generate_rows(params, computed_params, train_samples, batch_size, 1),
                               steps_per_epoch=nb_train_patterns // batch_size,
                               epochs=1000 if use_early_stopping else params['epochs'],
                               verbose=verbose,
                               validation_data=generate_rows(params, computed_params, val_samples, batch_size, 1),
                               validation_steps=nb_valid_patterns // batch_size,
                               callbacks=callbacks)
    print('max val_acc={}'.format(max(hist.history['val_acc'])))

    if use_early_stopping:
        #model.load_weights(weights_path)
        print('Restoring model weights from "{}"...'.format(weights_path))
        early_stopping.saver.restore(sess, weights_path)


def score_model(model, params, computed_params, val_samples):
    # прогоним валидационные паттерны через модель, чтобы получить f1 score.
    nb_valid_patterns = len(val_samples)
    v = next(generate_rows(params, computed_params, val_samples, nb_valid_patterns, 1))
    x = v[0]
    y_val = v[1]['output']

    assert(len(y_val) == nb_valid_patterns)

    y_pred = model.predict(x)
    f1 = sklearn.metrics.f1_score(y_true=y_val, y_pred=(y_pred >= 0.5).astype(np.int))
    #print('val f1={}'.format(f1))
    return f1

    #score = 0.0
    #try:
    #    score = sklearn.metrics.roc_auc_score(y_true=y_val, y_score=y_pred)
    #except ValueError as e:
    #    print(e)
    #    score = 0.0
    #
    #return score


def calc_rank_metrics(model, params, computed_params, val_samples, samples):
    phrase2samples_1 = dict()  # релевантные сэмплы для предложения
    phrase2samples_0 = dict()  # нерелевантные сэмплы для предложения
    for s in samples:
        if s.label == 1:
            if s.phrase1 in phrase2samples_1:
                phrase2samples_1[s.phrase1].add(s)
            else:
                phrase2samples_1[s.phrase1] = set([s])
        else:
            if s.phrase1 in phrase2samples_0:
                phrase2samples_0[s.phrase1].add(s)
            else:
                phrase2samples_0[s.phrase1] = set([s])

    rank_num = 0
    rank_denom = 0

    nb_good = 0  # сколько раз релевантный сэмпл был на первом месте
    nb_total = 0

    for val_sample in val_samples:
        if val_sample.label == 1 and val_sample.phrase1 != val_sample.phrase2:
            samples2 = [val_sample]

            phrase1 = val_sample.phrase1

            phrases2 = set()

            # Добавим готовые негативные примеры для phrase1
            if phrase1 in phrase2samples_0:
                phrases2.update(s.phrase2 for s in phrase2samples_0[phrase1])
                samples2.extend(phrase2samples_0[phrase1])

            # Добавим еще рандомных негативных сэмплов, чтобы общий размер списка на проверку был заданный.
            for s in sorted(samples, key=lambda _: random.random()):
                if s not in phrase2samples_1[phrase1] and s.phrase2 != phrase1:
                    if s.phrase2 not in phrases2:
                        phrases2.add(s.phrase2)
                        samples2.append(Sample(val_sample.phrase1, val_sample.words1, s.phrase2, s.words2, 0))
                        if len(phrases2) >= 100:
                            break

            x, _, used_samples = next(generate_rows(params, computed_params, samples2, len(samples2), 2))
            y_pred = model.predict(x)
            sample_rel = list((s, y[0]) for s, y in zip(used_samples, y_pred))
            sample_rel = sorted(sample_rel, key=lambda z: -z[1])

            # Ищем позицию для нашего сэмпла (у него label=1) в этом отсортированном списке
            rank = next(i for i, s in enumerate(sample_rel) if s[0].label == 1)
            rank_num += rank
            rank_denom += 1

            nb_total += 1
            if rank == 0:
                nb_good += 1

    return float(nb_good)/nb_total, rank_num/rank_denom


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='gridsearch', choices='train eval query gridsearch'.split())
    parser.add_argument('--tmp', type=str, default='../../../tmp')
    parser.add_argument('--dataset', default='../../../data/synonymy_dataset.csv')

    args = parser.parse_args()
    tmp_dir = args.tmp
    run_mode = args.run_mode
    dataset_path = args.dataset

    weights_path = os.path.abspath(os.path.join(tmp_dir, 'train_synonymy_detector_elmo.weights'))
    arch_path = os.path.abspath(os.path.join(tmp_dir, 'train_synonymy_detector_elmo.arch'))
    config_path = os.path.abspath(os.path.join(tmp_dir, 'train_synonymy_detector_elmo.config'))

    # файл в формате json с найденными оптимальными параметрами классификатора,
    # создается в ходе gridsearch, используется для train и eval
    best_params_path = os.path.join(tmp_dir, 'train_synonymy_detector_elmo.best_params.json')

    if run_mode == 'gridsearch':
        samples, computed_params = load_data(dataset_path, 10000)
        ys = [s.label for s in samples]

        best_params = None
        best_score = 0.0
        crossval_count = 0

        for epochs in [5]:
            for batch_size in [100]:
                for optimizer in ['nadam']:  # 'rmsprop', 'adam',
                    for units1 in [100]:
                        for activ1 in ['relu']:
                            for units2 in [0]:
                                for activ2 in ['relu']:
                                    model_params = {'optimizer': optimizer,
                                                    'batch_size': batch_size,
                                                    'epochs': epochs,
                                                    'units1': units1,
                                                    'units2': units2,
                                                    'activ1': activ1,
                                                    'activ2': activ2,
                                                    }

                                    crossval_count += 1
                                    kf = StratifiedKFold(n_splits=NFOLD)
                                    scores = []
                                    for ifold, (train_index, val_index) in enumerate(kf.split(samples, ys)):
                                        print('KFold[{}]'.format(ifold))
                                        train_samples = [samples[i] for i in train_index]
                                        val_samples = [samples[i] for i in val_index]

                                        model = create_model(model_params, computed_params)

                                        with tf.Session() as sess:
                                            tf.keras.backend.set_session(sess)
                                            tf.compat.v1.tables_initializer().run()

                                            train_model(model, model_params, computed_params, train_samples, val_samples, use_early_stopping=False, verbose=2)
                                            score = score_model(model, model_params, computed_params, val_samples)
                                            scores.append(score)

                                    score = np.mean(scores)
                                    score_std = np.std(scores)
                                    print('Crossvalidation #{} score={} std={}'.format(crossval_count, score, score_std))

                                    if score > best_score:
                                        print('!!! NEW BEST !!! score={} for {}'.format(score, get_params_str(model_params)))
                                        best_score = score
                                        best_params = model_params
                                        with open(best_params_path, 'w') as f:
                                            json.dump(best_params, f, indent=4)
                                    else:
                                        print('No improvement over current best_score={}'.format(best_score))

        print('best_score={} params={}'.format(best_score, get_params_str(best_params)))

    if run_mode == 'eval':
        samples, computed_params = load_data(dataset_path, 1000000)
        ys = [s.label for s in samples]

        with open(best_params_path, 'r') as f:
            model_params = json.load(f)

        kf = StratifiedKFold(n_splits=NFOLD)
        scores = []
        mean_poses = []
        for ifold, (train_index, val_index) in enumerate(kf.split(samples, ys)):
            print('KFold[{}]'.format(ifold))
            train_samples = [samples[i] for i in train_index]
            val_samples = [samples[i] for i in val_index]

            model = create_model(model_params, computed_params)

            with tf.Session() as sess:
                tf.keras.backend.set_session(sess)
                tf.compat.v1.tables_initializer().run()

                train_model(model, model_params, computed_params, train_samples, val_samples, use_early_stopping=True,
                            verbose=2)
                #score = score_model(model, model_params, computed_params, val_samples)
                print('Calculating rank metrics for fold #{}...'.format(ifold))
                precision1, mean_pos = calc_rank_metrics(model, model_params, computed_params, val_samples, samples)
                print('precision@1={} mean pos={}'.format(precision1, mean_pos))
                scores.append(precision1)
                mean_poses.append(mean_pos)

        score = np.mean(scores)
        score_std = np.std(scores)
        print('Crossvalidation precision@1={} std={} mean pos={}'.format(score, score_std, np.mean(mean_poses)))

    if run_mode == 'train':
        samples, computed_params = load_data(dataset_path, 10000)
        with open(best_params_path, 'r') as f:
            model_params = json.load(f)

        model = create_model(model_params, computed_params)
        with tf.Session() as sess:
            tf.keras.backend.set_session(sess)
            tf.compat.v1.tables_initializer().run()

            train_samples, val_samples = train_test_split(samples, test_size=0.1)
            train_model(model, model_params, computed_params, train_samples, val_samples, use_early_stopping=True, verbose=2)
            score = score_model(model, model_params, computed_params, val_samples)
            print('f1 score={}'.format(score))

            print('Calculating rank metrics...')
            precision1, mean_pos = calc_rank_metrics(model, model_params, computed_params, val_samples, samples)
            print('precision@1={} mean pos={}'.format(precision1, mean_pos))

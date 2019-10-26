# coding: utf-8
"""
Тренер детектора синонимичности двух фраз с использованием эмбеддингов BERT.
19.10.2019 первая реализация
22.10.2019 сохранение весом модели переделано на tf.train.Saver, так как керасовский метод не годится для графа с BERT
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
import tensorflow_hub as hub

# Установка: pip3 install bert-tensorflow
from bert.tokenization import FullTokenizer

from sklearn.model_selection import KFold
import sklearn.metrics
from sklearn.model_selection import train_test_split

import tensorflow.keras.callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import add, multiply, concatenate
from tensorflow.keras.models import Model

from bert_keras_layer import BertLayer


# Initialize session
sess = tf.Session()

NFOLD = 3
max_seq_length = 40
weights_path = None


def get_params_str(model_params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in model_params.items())


def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    #return FullTokenizer(vocab_file=os.path.join(bert_path, 'vocab.txt'), do_lower_case=False)


def convert_single_example(tokenizer, example, max_seq_length):
    if example is None:
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        return input_ids, input_mask, segment_ids

    tokens_a = tokenizer.tokenize(example)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


class Sample:
    def __init__(self):
        self.phrase1 = None
        self.phrase2 = None

        self.input_ids_1 = None
        self.input_mask_1 = None
        self.segment_ids_1 = None

        self.input_ids_2 = None
        self.input_mask_2 = None
        self.segment_ids_2 = None

        self.label = None


def load_data(bert_path, dataset_path, max_samples):
    computed_params = dict()

    samples = []
    max_len = 0

    tokenizer = create_tokenizer_from_hub_module(bert_path)

    df = pd.read_csv(dataset_path, encoding='utf-8', delimiter='\t', quoting=3)
    for i, r in df.iterrows():
        label = r['relevance']

        sample = Sample()
        sample.phrase1 = r['premise']
        sample.phrase2 = r['question']

        input_ids, input_mask, segment_ids = convert_single_example(tokenizer, r['premise'], max_seq_length)
        sample.input_ids_1 = input_ids
        sample.input_mask_1 = input_mask
        sample.segment_ids_1 = segment_ids

        input_ids, input_mask, segment_ids = convert_single_example(tokenizer, r['question'], max_seq_length)
        sample.input_ids_2 = input_ids
        sample.input_mask_2 = input_mask
        sample.segment_ids_2 = segment_ids

        sample.label = label
        samples.append(sample)

        max_len = max(max_len, len(sample.input_ids_1), len(sample.input_ids_2))
        if len(samples) >= max_samples:
            break

    computed_params['max_len'] = max_len

    if len(samples) > max_samples:
        samples = sorted(samples, key=lambda _: random.random())[:max_samples]

    return samples, computed_params



def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


def create_model(model_params, computed_params):
    global tables_initialized
    max_len = computed_params['max_len']
    bert_dim = 768

    in_id_1 = tf.keras.layers.Input(shape=(max_len,), name="input_ids_1")
    in_mask_1 = tf.keras.layers.Input(shape=(max_len,), name="input_masks_1")
    in_segment_1 = tf.keras.layers.Input(shape=(max_len,), name="segment_ids_1")

    in_id_2 = tf.keras.layers.Input(shape=(max_len,), name="input_ids_2")
    in_mask_2 = tf.keras.layers.Input(shape=(max_len,), name="input_masks_2")
    in_segment_2 = tf.keras.layers.Input(shape=(max_len,), name="segment_ids_2")

    bert_encoder = BertLayer(n_fine_tune_layers=1, bert_path=bert_path)
    encoder1 = bert_encoder([in_id_1, in_mask_1, in_segment_1])
    encoder2 = bert_encoder([in_id_2, in_mask_2, in_segment_2])

    addition = add([encoder1, encoder2])
    minus_y1 = Lambda(lambda x: -x, output_shape=(bert_dim,))(encoder1)
    mul = add([encoder2, minus_y1])
    mul = multiply([mul, mul])
    encoder = tensorflow.keras.layers.concatenate(inputs=[mul, addition, encoder1, encoder2])

    #encoder = encoder1
    #encoder = tensorflow.keras.layers.concatenate(inputs=[encoder1, encoder2])

    net = encoder
    net = Dense(units=model_params['units1'], activation=model_params['activ1'])(net)
    if model_params['units2'] > 0:
        net = Dense(units=model_params['units2'], activation=model_params['activ2'])(net)

    net = Dense(units=1, name='output', activation='sigmoid')(net)

    model = Model(inputs=[in_id_1, in_mask_1, in_segment_1, in_id_2, in_mask_2, in_segment_2], outputs=net)
    model.compile(loss='binary_crossentropy', optimizer=model_params['optimizer'], metrics=['accuracy'])

    initialize_vars(sess)

    return model


def generate_rows(params, computed_params, samples, batch_size, mode):
    batch_index = 0
    batch_count = 0
    max_len = computed_params['max_len']

    X1_1_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    X2_1_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    X3_1_batch = np.zeros((batch_size, max_len), dtype=np.int32)

    X1_2_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    X2_2_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    X3_2_batch = np.zeros((batch_size, max_len), dtype=np.int32)

    y_batch = np.zeros((batch_size), dtype=np.bool)

    while True:
        used_samples = []
        for irow, sample in enumerate(sorted(samples, key=lambda _: random.random())):
            used_samples.append(sample)
            X1_1_batch[batch_index, :] = sample.input_ids_1
            X2_1_batch[batch_index, :] = sample.input_mask_1
            X3_1_batch[batch_index, :] = sample.segment_ids_1

            X1_2_batch[batch_index, :] = sample.input_ids_2
            X2_2_batch[batch_index, :] = sample.input_mask_2
            X3_2_batch[batch_index, :] = sample.segment_ids_2

            y_batch[batch_index] = sample.label
            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                x_inputs = {'input_ids_1': X1_1_batch, 'input_masks_1': X2_1_batch, 'segment_ids_1': X3_1_batch,
                            'input_ids_2': X1_2_batch, 'input_masks_2': X2_2_batch, 'segment_ids_2': X3_2_batch
                            }

                if mode == 0:
                    yield x_inputs
                elif mode == 1:
                    yield (x_inputs, {'output': y_batch})
                else:
                    yield (x_inputs, {'output': y_batch}, used_samples)

                # очищаем матрицы порции для новой порции
                X1_1_batch.fill(0)
                X2_1_batch.fill(0)
                X3_1_batch.fill(0)
                X1_2_batch.fill(0)
                X2_2_batch.fill(0)
                X3_2_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0



class VisualizeCallback(tensorflow.keras.callbacks.Callback):
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

    callbacks = []

    #monitor_metric = 'val_acc'
    #if True:
    #    model_checkpoint = ModelCheckpoint(weights_path,
    #                                       monitor=monitor_metric,
    #                                       verbose=1,
    #                                       save_best_only=True,
    #                                       mode='auto')
    #    callbacks.append(model_checkpoint)

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
    # прогоним валидационные паттерны через модель, чтобы получить финальную оценку качества.
    nb_valid_patterns = len(val_samples)
    for v in generate_rows(params, computed_params, val_samples, nb_valid_patterns, 1):
        x = v[0]
        y_val = v[1]['output']
        break

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

    tokenizer = create_tokenizer_from_hub_module(bert_path)

    rank_num = 0
    rank_denom = 0

    for val_sample in val_samples:
        if val_sample.label == 1 and val_sample.phrase1 != val_sample.phrase2:
            phrase1 = val_sample.phrase1

            # Для предложения phrase1 подберем список сопоставляемых предложений, среди которых
            # будет только одно релевантное, а остальные - рандомные нерелевантные
            phrases2 = set([val_sample.phrase2])

            # Добавим готовые негативные примеры для phrase1
            if phrase1 in phrase2samples_0:
                phrases2.update(s.phrase2 for s in phrase2samples_0[phrase1])

            # Добавим еще рандомных негативных сэмплов, чтобы общий размер списка на проверку был заданный.
            for s in sorted(samples, key=lambda _: random.random()):
                if s not in phrase2samples_1[phrase1] and s.phrase2 != phrase1:
                    phrases2.add(s.phrase2)
                    if len(phrases2) >= 100:
                        break

            samples2 = []
            for phrase2 in phrases2:
                sample = Sample()
                sample.phrase1 = phrase1
                sample.phrase2 = phrase2

                input_ids, input_mask, segment_ids = convert_single_example(tokenizer, phrase1, max_seq_length)
                sample.input_ids_1 = input_ids
                sample.input_mask_1 = input_mask
                sample.segment_ids_1 = segment_ids

                input_ids, input_mask, segment_ids = convert_single_example(tokenizer, phrase2, max_seq_length)
                sample.input_ids_2 = input_ids
                sample.input_mask_2 = input_mask
                sample.segment_ids_2 = segment_ids

                samples2.append(sample)

            for v in generate_rows(params, computed_params, samples2, len(samples2), 2):
                x = v[0]
                used_samples = v[2]
                break

            y_pred = model.predict(x)
            sample_rel = list((s, y[0]) for s, y in zip(used_samples, y_pred))
            sample_rel = sorted(sample_rel, key=lambda z: -z[1])

            # Ищем позицию для нашего сэмпла (у него label=1) в этом отсортированном списке
            rank = next(i for i, s
                        in enumerate(sample_rel)
                        if s[0].phrase1 == val_sample.phrase1 and s[0].phrase2 == val_sample.phrase2)
            rank_num += rank
            rank_denom += 1

    return rank_num / rank_denom


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='train', choices='train query gridsearch'.split())
    parser.add_argument('--tmp', type=str, default='../../../tmp')
    parser.add_argument('--dataset', default='../../../data/synonymy_dataset.csv')

    args = parser.parse_args()
    tmp_dir = args.tmp
    run_mode = args.run_mode
    dataset_path = args.dataset

    # Гугловский multilingual:
    bert_path = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"

    # deeppavlov ruBERT:
    #bert_path = '/mnt/7383b08e-ace3-49d3-8991-5b9aa07d2596/EmbeddingModels/BERT_multilingual/model/rubert_cased_L-12_H-768_A-12_v1'

    weights_path = os.path.abspath(os.path.join(tmp_dir, 'train_synonymy_detector_bert_checkpoints/chkpoint'))
    arch_path = os.path.abspath(os.path.join(tmp_dir, 'train_synonymy_detector_bert.arch'))
    config_path = os.path.abspath(os.path.join(tmp_dir, 'train_synonymy_detector_bert.config'))

    if run_mode == 'gridsearch':
        # Для ускорения перебора возьмем небольшую часть полного датасета
        samples, computed_params = load_data(bert_path, dataset_path, 10000)

        best_params = None
        best_score = 0.0
        crossval_count = 0

        for epochs in [10, 50]:
            for batch_size in [50]:
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
                                    kf = KFold(n_splits=NFOLD)
                                    scores = []
                                    for ifold, (train_index, val_index) in enumerate(kf.split(samples)):
                                        print('KFold[{}]'.format(ifold))
                                        train_samples = [samples[i] for i in train_index]
                                        val_samples = [samples[i] for i in val_index]

                                        model = create_model(model_params, computed_params)
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
                                    else:
                                        print('No improvement over current best_score={}'.format(best_score))

        print('best_score={} params={}'.format(best_score, get_params_str(best_params)))

    if run_mode == 'train':
        samples, computed_params = load_data(bert_path, dataset_path, 1000000)

        model_params = {'optimizer': 'nadam',
                        'batch_size': 100,
                        'units1': 50,
                        'units2': 0,
                        'activ1': 'sigmoid',
                        'activ2': 'sigmoid',
                        }

        model = create_model(model_params, computed_params)
        train_samples, val_samples = train_test_split(samples, test_size=0.1)
        train_model(model, model_params, computed_params, train_samples, val_samples, use_early_stopping=True, verbose=2)
        score = score_model(model, model_params, computed_params, val_samples)
        print('f1 score={}'.format(score))

        print('Calculating rank metrics...')
        rank_score = calc_rank_metrics(model, model_params, computed_params, val_samples, samples)
        print('rank_score={}'.format(rank_score))

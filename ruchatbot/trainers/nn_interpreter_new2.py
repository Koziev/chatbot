# coding: utf-8
"""
Тренер классификатора шаблонов интерпретации для чатбота
03.08.2019 первая реализация
03.08.2019 добавлен вариант архитектуры CRF
15.09.2019 переход на word_embeddings.WordEmbeddings, чтобы чатбот мог использовать KeyedVectors.load(mmap='r')
"""

from __future__ import print_function
import numpy as np
import argparse
import io
import os
import json
import itertools
import logging

from sklearn.model_selection import cross_val_score
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

import keras.callbacks
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import recurrent
from keras.layers import Dropout
from keras.layers.core import RepeatVector
from keras.layers.core import Dense
from keras.layers.merge import concatenate, add, multiply
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
import keras.regularizers
from keras.wrappers.scikit_learn import KerasClassifier

import keras_contrib
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import keras_contrib.optimizers

from ruchatbot.utils.tokenizer import Tokenizer
import ruchatbot.utils.console_helpers
import ruchatbot.utils.logging_helpers
from word_embeddings import WordEmbeddings


def get_params_str(model_params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in model_params.items())


def prepare_phrase(phrase):
    for delim in u'?,!«»"()':
        phrase = phrase.replace(delim, ' ' + delim + ' ')

    if phrase[-1] == '.':
        phrase = phrase[:-1]
    phrase = phrase.replace('  ', ' ').strip()
    return phrase


def instance_accuracy(y_true, y_pred):
    nb_good = sum(np.array_equal(y1, y2) for (y1, y2) in zip(y_true, y_pred))
    nb_total = len(y_true)
    return float(nb_good) / nb_total


class Sample(object):
    def __init__(self, question, short_answer, expanded_answer, template):
        self.question = question
        self.short_answer = short_answer
        self.expanded_answer = expanded_answer
        self.template = template
        self.question_words = None
        self.short_answer_words = None
        self.expanded_answer_words = None


PAD_WORD = u''
BEG_TOKEN = '<begin>'
END_TOKEN = '<end>'


def load_data(dataset_path, tokenizer):
    samples = []
    max_inputseq_len = 0
    max_outputseq_len = 0
    all_labels = set()
    all_terms = set([BEG_TOKEN, END_TOKEN, PAD_WORD])
    logging.info('Loading dataset "%s"', dataset_path)
    with io.open(dataset_path, 'r', encoding='utf-8') as rdr:
        for line in rdr:
            tx = line.strip().split('\t')
            all_labels.add(tx[3])
            question_words = tokenizer.tokenize(tx[0])
            short_answer_words = tokenizer.tokenize(tx[1])
            expanded_answer_words = tokenizer.tokenize(tx[2])
            max_inputseq_len = max(max_inputseq_len, len(question_words), len(short_answer_words))
            max_outputseq_len = max(max_outputseq_len, len(expanded_answer_words))
            sample = Sample(tx[0], tx[1], tx[2], tx[3])
            sample.question_words = question_words
            sample.short_answer_words = short_answer_words
            sample.expanded_answer_words = expanded_answer_words
            samples.append(sample)

            terms = tx[3].split()
            all_terms.update(terms)

    label2index = dict((l, i) for i, l in enumerate(all_labels))
    term2index = dict((t, i) for i, t in enumerate(all_terms))

    computed_params = {'max_inputseq_len': max_inputseq_len,
                       'max_outputseq_len': max_outputseq_len,
                       'label2index': label2index,
                       'nb_labels': len(all_labels),
                       'term2index': term2index,
                       'nb_terms': len(all_terms)}

    return samples, computed_params


def lpad_wordseq(words, n):
    """ Слева добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


def rpad_wordseq(words, n):
    """ Справа добавляем пустые слова, чтобы длина строки words стала равна n """
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def pad_wordseq(words, n, padding):
    if padding == 'right':
        return rpad_wordseq(words, n)
    else:
        return lpad_wordseq(words, n)


def pad_for_crf(terms, max_len):
    terms2 = [BEG_TOKEN] + terms + [END_TOKEN] + [PAD_WORD]*(max_len-len(terms)-2)
    return terms2


def load_embeddings(wordchar2vector_path, word2vector_path, computed_params):
    embeddings = WordEmbeddings.load_word_vectors(wordchar2vector_path, word2vector_path)
    computed_params['word_dims'] = embeddings.get_vector_size()
    computed_params['word2vec'] = embeddings
    return embeddings


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate( words ):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def vectorize_samples(samples, params, computed_params):
    padding = params['padding']
    nb_samples = len(samples)
    max_inputseq_len = computed_params['max_inputseq_len']
    max_outputseq_len = computed_params['max_outputseq_len']
    word_dims = computed_params['word_dims']
    w2v = computed_params['word2vec']
    nb_labels = computed_params['nb_labels']
    label2index = computed_params['label2index']
    term2index = computed_params['term2index']
    nb_terms = computed_params['nb_terms']

    if params['arch'] == 'bilstm':
        X1_data = np.zeros((nb_samples, max_inputseq_len, word_dims), dtype=np.float32)
        X2_data = np.zeros((nb_samples, max_inputseq_len, word_dims), dtype=np.float32)
        y_data = np.zeros((nb_samples, nb_labels), dtype=np.bool)

        for isample, sample in enumerate(samples):
            words1 = pad_wordseq(sample.question_words, max_inputseq_len, padding)
            vectorize_words(words1, X1_data, isample, w2v)

            words2 = pad_wordseq(sample.short_answer_words, max_inputseq_len, padding)
            vectorize_words(words2, X2_data, isample, w2v)

            y_data[isample, label2index[sample.template]] = 1
    elif params['arch'] == 'crf':
        max_len = max(max_inputseq_len, max_outputseq_len)+2

        X1_data = np.zeros((nb_samples, max_len, word_dims), dtype=np.float32)
        X2_data = np.zeros((nb_samples, max_len, word_dims), dtype=np.float32)
        y_data = np.zeros((nb_samples, max_len, nb_terms), dtype=np.bool)

        for isample, sample in enumerate(samples):
            words1 = pad_wordseq(sample.question_words, max_len, padding)
            vectorize_words(words1, X1_data, isample, w2v)

            words2 = pad_wordseq(sample.short_answer_words, max_len, padding)
            vectorize_words(words2, X2_data, isample, w2v)

            for iterm, term in enumerate(pad_for_crf(sample.template.split(), max_len)):
                y_data[isample, iterm, term2index[term]] = 1

    return X1_data, X2_data, y_data


def create_model(computed_params, model_params):
    max_inputseq_len = computed_params['max_inputseq_len']
    max_outputseq_len = computed_params['max_outputseq_len']

    nb_labels = computed_params['nb_labels']
    word_dims = computed_params['word_dims']

    if model_params['arch'] == 'crf':
        max_len = max(max_inputseq_len, max_outputseq_len) + 2
        input1 = Input(shape=(max_len, word_dims,), dtype='float32', name='input1')
        input2 = Input(shape=(max_len, word_dims,), dtype='float32', name='input2')
    else:
        input1 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input1')
        input2 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input2')

    optimizer = model_params['optimizer']
    arch = model_params['arch']
    if arch == 'bilstm':
        net1 = Bidirectional(recurrent.LSTM(units=model_params['rnn_size'],
                                            dropout=model_params['dropout_rate'],
                                            return_sequences=False))(input1)

        net2 = Bidirectional(recurrent.LSTM(units=model_params['rnn_size'],
                                            dropout=model_params['dropout_rate'],
                                            return_sequences=False))(input2)

        net = concatenate([net1, net2])
    elif arch in ('crf', 'rnn_seq'):
        net1 = Bidirectional(recurrent.LSTM(units=model_params['rnn_size'],
                                            dropout=model_params['dropout_rate'],
                                            return_sequences=False))(input1)

        net2 = Bidirectional(recurrent.LSTM(units=model_params['rnn_size'],
                                            dropout=model_params['dropout_rate'],
                                            return_sequences=False))(input2)

        net = concatenate([net1, net2])
    else:
        raise NotImplementedError()

    if model_params['dense1'] > 0:
        net = Dense(units=model_params['dense1'], activation='sigmoid')(net)

    if arch == 'crf':
        net = RepeatVector(max_len)(net)
        net = recurrent.LSTM(model_params['rnn_size'], return_sequences=True)(net)
        net = CRF(units=computed_params['nb_terms'], sparse_target=False)(net)
        model = Model(inputs=[input1, input2], outputs=net)
        model.compile(loss=crf_loss, optimizer=optimizer, metrics=[crf_viterbi_accuracy])
    elif arch == 'bilstm':
        net = Dense(units=nb_labels, activation='softmax')(net)
        model = Model(inputs=[input1, input2], outputs=net)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='gridsearch', choices='train query gridsearch'.split())
    parser.add_argument('--tmp', type=str, default='../tmp')
    parser.add_argument('--dataset', default='../tmp/interpreter_templates.tsv')
    parser.add_argument('--wordchar2vector', type=str, default='../tmp/wc2v.kv', help='path to wordchar2vector model dataset')
    parser.add_argument('--word2vector', type=str, default='../tmp/w2v.kv', help='path to word2vector model file')

    args = parser.parse_args()
    tmp_dir = args.tmp
    run_mode = args.run_mode
    dataset_path = args.dataset

    wordchar2vector_path = args.wordchar2vector
    word2vector_path = os.path.expanduser(args.word2vector)

    # настраиваем логирование в файл
    ruchatbot.utils.logging_helpers.init_trainer_logging(os.path.join(tmp_dir, 'nn_interpreter_new1.log'))
    logging.debug('Started run_mode=%s', run_mode)

    weights_file = os.path.abspath(os.path.join(tmp_dir, 'nn_interpreter_new1.weights'))
    arch_file = os.path.abspath(os.path.join(tmp_dir, 'nn_interpreter_new1.arch'))
    config_file = os.path.abspath(os.path.join(tmp_dir, 'nn_interpreter_new1.config'))

    best_path = os.path.join(tmp_dir, 'nn_interpreter_new1.best_params.json')

    tokenizer = ruchatbot.utils.tokenizer.Tokenizer()
    tokenizer.load()

    if run_mode == 'gridsearch':
        best_params = None
        best_score = 0.0

        samples, computed_params = load_data(dataset_path, tokenizer)
        load_embeddings(wordchar2vector_path, word2vector_path, computed_params)
        sample_labels = [sample.template for sample in samples]

        model_params = dict()
        crossval_count = 0

        for padding in ['right']:
            model_params['padding'] = padding
            for arch in ['crf']:  # 'rnn_seq', 'crf', 'bilstm'
                model_params['arch'] = arch
                X1_data, X2_data, y_data = vectorize_samples(samples, model_params, computed_params)

                for batch_size in [50]:
                    for epochs in [40, 50, 60]:
                        for optimizer in ['nadam']:  # 'rmsprop', 'adam',
                            for rnn_size in [100, 200]:
                                for dropout_rate in [0.0]:
                                    for dense1 in [0]:
                                        model_params['batch_size'] = batch_size
                                        model_params['epochs'] = epochs
                                        model_params['padding'] = padding
                                        model_params['optimizer'] = optimizer
                                        model_params['rnn_size'] = rnn_size
                                        model_params['dropout_rate'] = dropout_rate
                                        model_params['dense1'] = dense1

                                        kf = StratifiedKFold(n_splits=5)
                                        scores = []
                                        crossval_count += 1
                                        for ifold, (train_index, test_index) in enumerate(kf.split(samples, sample_labels)):
                                            logging.info('KFold[%d]', ifold)
                                            X1_train = X1_data[train_index]
                                            X2_train = X2_data[train_index]
                                            y_train = y_data[train_index]

                                            X1_test = X1_data[test_index]
                                            X2_test = X2_data[test_index]
                                            y_test = y_data[test_index]

                                            model = create_model(computed_params, model_params)
                                            model.fit({'input1': X1_train, 'input2': X2_train}, y_train,
                                                      batch_size=batch_size, epochs=epochs, verbose=2)
                                            y_pred = model.predict({'input1': X1_test, 'input2': X2_test}, verbose=0)

                                            max_len = max(computed_params['max_inputseq_len'], computed_params['max_outputseq_len']) + 2
                                            y_true2 = np.argmax(y_test, axis=-1).reshape(len(y_test)*max_len)
                                            y_pred2 = np.argmax(y_pred, axis=-1).reshape(len(y_test)*max_len)

                                            #score = sklearn.metrics.accuracy_score(y_true=y_true2, y_pred=y_pred2)
                                            score = instance_accuracy(y_true=np.argmax(y_test, axis=-1), y_pred=np.argmax(y_pred, axis=-1))
                                            logging.info('KFold[%d] score=%f', ifold, score)

                                            scores.append(score)

                                        score = np.mean(scores)
                                        score_std = np.std(scores)
                                        logging.info('Crossvalidation #%d score=%f std=%f', crossval_count, score, score_std)

                                        if score > best_score:
                                            logging.info('!!! NEW BEST !!! score=%f for %s', score, get_params_str(model_params))
                                            best_score = score
                                            best_params = model_params.copy()
                                            with open(best_path, 'w') as f:
                                                json.dump(best_params, f, indent=4)
                                        else:
                                            logging.info('No improvement over current best_score=%f', best_score)

        logging.info('All done, best_score=%f params=%s', best_score, get_params_str(best_params))
    elif run_mode == 'train':
        with open(best_path, 'r') as f:
            model_params = json.load(f)
        logging.info('Will train using params: %s', get_params_str(model_params))

        samples, computed_params = load_data(dataset_path, tokenizer)
        load_embeddings(wordchar2vector_path, word2vector_path, computed_params)

        X1_data, X2_data, y_data = vectorize_samples(samples, model_params, computed_params)

        epochs = model_params['epochs']
        batch_size = model_params['batch_size']

        model = create_model(computed_params, model_params)
        with open(arch_file, 'w') as f:
            f.write(model.to_json())

        model.fit({'input1': X1_data, 'input2': X2_data}, y_data, epochs=epochs, batch_size=batch_size, verbose=2)
        model.save_weights(weights_file)

        config = {'w2v_path': word2vector_path,
                  'index2label': [(i, l) for (l, i) in computed_params['label2index'].items()],
                  'index2term': [(i, t) for (t, i) in computed_params['term2index'].items()],
                  'weights': weights_file,
                  'arch_file': arch_file,
                  }
        config.update(model_params)
        config['max_inputseq_len'] = computed_params['max_inputseq_len']
        config['max_outputseq_len'] = computed_params['max_outputseq_len']
        config['label2index'] = computed_params['label2index']
        config['nb_labels'] = computed_params['nb_labels']
        config['term2index'] = computed_params['term2index']
        config['nb_terms'] = computed_params['nb_terms']
        config['word_dims'] = computed_params['word_dims']

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    elif run_mode == 'query':
        with open(config_file, 'r') as f:
            model_config = json.load(f)
            index2label = dict(model_config['index2label'])
            index2term = dict(model_config['index2term'])
            arch_file = model_config['arch_file']
            weights_file = model_config['weights']

        with open(arch_file, 'r') as f:
            model = model_from_json(f.read(), {'CRF': CRF})

        model.load_weights(weights_file)

        computed_params = model_config.copy()
        load_embeddings(wordchar2vector_path, word2vector_path, computed_params)

        while True:
            question = input('Q:> ').strip()
            question = prepare_phrase(question)

            short_answer = input('A:> ').strip()
            short_answer = prepare_phrase(short_answer)

            sample = Sample(question, short_answer, '', '')
            sample.question_words = tokenizer.tokenize(question)
            sample.short_answer_words = tokenizer.tokenize(short_answer)
            samples = [sample]
            X1_data, X2_data, y_data = vectorize_samples(samples, model_config, computed_params)

            y_pred = model.predict({'input1': X1_data, 'input2': X2_data}, verbose=0)
            if model_config['arch'] == 'bilstm':
                label = index2label[np.argmax(y_pred[0])]
                print(u'template={}'.format(label))
            elif model_config['arch'] == 'crf':
                terms = np.argmax(y_pred[0], axis=-1)
                terms = [index2term[i] for i in terms]
                print('{}\n\n'.format(u' '.join(terms)))
            else:
                raise NotImplementedError()

"""
Тренировка модели для интерпретации фраз с помощью seq2seq и attention.
Для чатбота.
"""

import io
import os
import itertools
import random
import json
import argparse

import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.model_selection import KFold

import sentencepiece as spm
from colorclass import Color, Windows
import terminaltables

import keras
import keras.callbacks
from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


# https://github.com/asmekal/keras-monotonic-attention
from ruchatbot.layers.attention_decoder import AttentionDecoder



def get_params_str(model_params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in model_params.items())


def dress_context_line(s):
    if s[-1] in '.?!':
        return s
    else:
        return s + ' .'


def undress_output_line(s):
    if s[-1] in '.?!':
        return s[:-1].strip()
    else:
        return s


def train_bpe_model(params):
    spm_items = params['spm_items']

    # Готовим корпус для обучения SentencePiece
    sentencepiece_corpus = os.path.join(tmp_dir, 'sentencepiece_corpus.txt')

    nb_samples = 0
    #max_nb_samples = 10000000  # макс. кол-во предложений для обучения SentencePiece
    with io.open(sentencepiece_corpus, 'w', encoding='utf-8') as wrt:
        df = pd.read_csv(os.path.join(tmp_dir, 'interpreter_samples.tsv'), delimiter='\t')
        for i, r in df.iterrows():
            left_data = r.context
            right_data = r.output
            wrt.write('{}\n'.format(left_data))
            wrt.write('{}\n'.format(right_data))
            nb_samples += 1

    spm_name = 'nn_seq2seq_interpreter.sentencepiece'

    if not os.path.exists(os.path.join(tmp_dir, spm_name + '.vocab')):
        print('Start training bpe model "{}" on {} samples'.format(spm_name, nb_samples))
        spm.SentencePieceTrainer.Train(
            '--input={} --model_prefix={} --vocab_size={} --shuffle_input_sentence=true --character_coverage=1.0 --model_type=unigram'.format(
                sentencepiece_corpus, spm_name, spm_items))
        os.rename(spm_name + '.vocab', os.path.join(tmp_dir, spm_name + '.vocab'))
        os.rename(spm_name + '.model', os.path.join(tmp_dir, spm_name + '.model'))

    print('bpe model "{}" ready'.format(spm_name))
    return spm_name


def load_bpe_model(spm_name):
    sp = spm.SentencePieceProcessor()
    rc = sp.Load(os.path.join(tmp_dir, spm_name + '.model'))
    print('bpe model "{}" loaded with status={}'.format(spm_name, rc))
    return sp


class Sample:
    def __init__(self):
        self.left_str = None
        self.left_tokens = None
        self.segments = None  # для каждого токена тут будет номер фразы (0 - последняя, 1 - предпоследняя etc)
        self.right_str = None
        self.right_tokens = None


def create_prediction_sample(bpe_model, context):
    sample = Sample()

    sample.left_str = context
    sample.left_tokens = bpe_model.EncodeAsPieces(sample.left_str)

    nb_segm = sample.left_str.count('|')
    isegm = nb_segm
    sample.segments = []
    for token in sample.left_tokens:
        sample.segments.append(isegm)
        if token == '|':
            isegm -= 1

    sample.right_str = ''
    sample.right_tokens = []
    return sample


def load_samples(bpe_model, computed_params, max_samples):
    all_tokens = set()
    samples = []
    df = pd.read_csv(os.path.join(tmp_dir, 'interpreter_samples.tsv'), delimiter='\t')
    max_segments = 0
    for i, r in df.iterrows():
        sample = Sample()

        # НАЧАЛО ОТЛАДКИ
        #r.context = 'тебя зовут денис , верно ? | да'
        #r.output = 'меня зовут денис'
        # КОНЕЦ ОТЛАДКИ

        sample.left_str = r.context
        #x = sample.left_str.rindex('|')
        #sample.left_str = sample.left_str[:x] + '#' + sample.left_str[x+1:]
        sample.left_tokens = bpe_model.EncodeAsPieces(sample.left_str)

        nb_segm = sample.left_str.count('|')
        max_segments = max(max_segments, nb_segm)
        isegm = nb_segm
        sample.segments = []
        for token in sample.left_tokens:
            sample.segments.append(isegm)
            if token == '|':
                isegm -= 1

        sample.right_str = r.output
        sample.right_tokens = bpe_model.EncodeAsPieces(sample.right_str)
        samples.append(sample)

        all_tokens.update(sample.left_tokens)
        all_tokens.update(sample.right_tokens)

        # НАЧАЛО ОТЛАДКИ
        #break
        # КОНЕЦ ОТЛАДКИ


    print('samples.count={}'.format(len(samples)))

    if len(samples) > max_samples:
        print('Shrinking datasate to {} samples...'.format(max_samples))
        samples = samples[:max_samples]

    print('all_tokens.count={}'.format(len(all_tokens)))
    token2index = dict((t, i) for i, t in enumerate(all_tokens, start=1))
    token2index[''] = 0

    max_left_len = max(map(len, (s.left_tokens for s in samples)))
    max_right_len = max(map(len, (s.right_tokens for s in samples)))
    print('max_left_len={}'.format(max_left_len))
    print('max_right_len={}'.format(max_right_len))

    computed_params['token2index'] = token2index
    computed_params['max_left_len'] = max_left_len
    computed_params['max_right_len'] = max_right_len
    computed_params['max_segments'] = max_segments+1

    return samples


def vectorize_samples(samples, params, computed_params):
    nb_samples = len(samples)

    token2index = computed_params['token2index']
    max_left_len = computed_params['max_left_len']
    max_right_len = computed_params['max_right_len']
    seq_len = max(max_left_len, max_right_len)

    X1 = np.zeros((nb_samples, seq_len), dtype=np.int32)
    encode_segments = params['encode_segments']
    if encode_segments:
        X2 = np.zeros((nb_samples, seq_len), dtype=np.int32)
        Xs = [X1, X2]
    else:
        Xs = [X1]

    y = np.zeros((nb_samples, seq_len), dtype=np.int32)
    for isample, sample in enumerate(samples):
        left_tokens = sample.left_tokens

        # нАЧАЛО ОТЛАДКИ - ВЫРАВНИВАЕМ ВПРАВО
        #left_tokens = ['']*(seq_len-len(left_tokens)) + left_tokens
        # КОНЕЦ ОТЛАДКИ

        for itoken, (token, isegm) in enumerate(zip(left_tokens, sample.segments)):
            if token in token2index:
                X1[isample, itoken] = token2index[token]

            if encode_segments:
                X2[isample, itoken] = isegm

        for itoken, token in enumerate(sample.right_tokens):
            if token in token2index:
                y[isample, itoken] = token2index[token]

    return Xs, y


def create_model(params, computed_params):
    seq_len = max(computed_params['max_left_len'], computed_params['max_right_len'])
    hidden_dim = params['hidden_dim']

    input_tokens = keras.layers.Input(shape=(seq_len,), dtype=np.int32, name='tokens')
    inputs = [input_tokens]

    net1 = Embedding(input_dim=params['spm_items'], output_dim=params['token_dim'], input_length=seq_len)(input_tokens)

    if params['encode_segments']:
        input_segments = keras.layers.Input(shape=(seq_len,), dtype=np.int32, name='segments')
        inputs.append(input_segments)
        net2 = Embedding(input_dim=computed_params['max_segments'], output_dim=8, input_length=seq_len)(input_segments)
        net = keras.layers.concatenate([net1, net2])
    else:
        net = net1

    net = Bidirectional(LSTM(hidden_dim, return_sequences=True))(net)
    f = params['is_monotonic']
    net = AttentionDecoder(units=hidden_dim, alphabet_size=params['spm_items'],
                           embedding_dim=params['token_dim'],
                           is_monotonic=f, normalize_energy=f)(net)

    if params['crf']:
        net = CRF(units=params['spm_items'], sparse_target=True)(net)

    #opt = keras.optimizers.Adadelta(lr=0.1)
    opt = keras.optimizers.Nadam(lr=0.002)  # 0.005
    #opt = keras.optimizers.Nadam()
    #opt = keras.optimizers.Adam(lr=0.005)
    #opt = keras.optimizers.Ftrl()

    model = Model(inputs=inputs, outputs=net)

    if params['crf']:
        model.compile(loss=crf_loss, optimizer=opt)  #, metrics=[crf_viterbi_accuracy])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)

    model.summary()
    return model


def jaccard(words1, words2):
    s1 = set(words1)
    s2 = set(words2)
    return float(len(s1&s2))/float(1e-8+len(s1|s2))


def score_model(model, samples, Xs, y, computed_params):
    index2token = dict((i, t) for t, i in computed_params['token2index'].items())
    y_pred = model.predict(Xs, verbose=0)
    y_pred = np.argmax(y_pred, axis=-1)
    sum_jac = 0.0
    denom = 0
    for sample, y_pred_sample in zip(samples, y_pred):
        # Декодируем список индексов предсказанных токенов
        tokens = [index2token.get(itok, '[???]') for itok in y_pred_sample]
        pred_right = ''.join(tokens).replace('▁', ' ').strip()
        pred_words = pred_right.split(' ')
        true_words = sample.right_str.split(' ')
        jac = jaccard(pred_words, true_words)
        sum_jac += jac
        denom += 1

    score = sum_jac / denom
    return score


class VizualizeCallback(keras.callbacks.Callback):
    """
    После каждой эпохи обучения делаем сэмплинг образцов из текущей модели,
    чтобы видеть общее качество.
    """

    def __init__(self, model, test_samples, params, computed_params, weights_path, patience):
        self.model = model
        self.model_params = params
        self.computed_params = computed_params
        self.test_samples = test_samples
        self.weights_path = weights_path
        self.Xs_test, self.y = vectorize_samples(test_samples, self.model_params, self.computed_params)
        self.index2token = dict((i, t) for t, i in computed_params['token2index'].items())
        self.best_score = -np.inf
        self.epoch = 0
        self.best_epoch = 0
        self.patience = patience
        self.wait = 0

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        print('Epoch {} validation...'.format(self.epoch))

        # Оценка текущего качества
        score = score_model(self.model, self.test_samples, self.Xs_test, self.y, self.computed_params)
        if score > self.best_score:
            print('NEW BEST SCORE={}'.format(score))
            self.best_score = score
            self.best_epoch = self.epoch
            self.wait = 0
            self.model.save_weights(self.weights_path, overwrite=True)

            # отберем немного сэмлов для визуализации текущего состояния модели
            # 16-06-2020 не будем показывать сэмплы с длиной левой части больше 71, чтобы не разваливались
            # ascii-таблицы.
            samples2 = sorted(filter(lambda s: len(s.left_str) < 72, test_samples), key=lambda z: random.random())[:10]
            Xs, y = vectorize_samples(samples2, self.model_params, self.computed_params)
            y_pred = model.predict(Xs, verbose=0)
            y_pred = np.argmax(y_pred, axis=-1)

            table = ['context true_output predicted_output'.split()]
            for sample, y_pred_sample in zip(samples2, y_pred):
                # Декодируем список индексов предсказанных токенов
                tokens = [self.index2token[itok] for itok in y_pred_sample]
                pred_right = ''.join(tokens).replace('▁', ' ').strip()

                true2 = undress_output_line(sample.right_str)
                pred2 = undress_output_line(pred_right)

                if pred2 == true2:
                    # выдача сетки полностью верная
                    output2 = Color('{autogreen}' + pred_right + '{/autogreen}')
                elif jaccard(pred2.split(), true2.split()) > 0.5:
                    # выдача сетки частично совпала с требуемой строкой
                    output2 = Color('{autoyellow}' + pred_right + '{/autoyellow}')
                else:
                    # неправильная выдача сетки
                    output2 = Color('{autored}' + pred_right + '{/autored}')

                table.append((sample.left_str, sample.right_str, output2))

            table = terminaltables.AsciiTable(table)
            print(table.table)
        else:
            print('val score={}; no improvement over best score={} at epoch={}'.format(score, self.best_score, self.best_epoch))
            self.wait += 1
            if self.wait > self.patience:
                print('Early stopping at epoch={} with best score={}'.format(self.epoch, self.best_score))
                self.model.stop_training = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model trainer for PQA answer generator')
    parser.add_argument('--run_mode', choices='gridsearch train query'.split(), default=None)
    parser.add_argument('--tmp_dir', default='../../tmp')
    parser.add_argument('--data_dir', default='../../data')
    args = parser.parse_args()

    while not args.run_mode:
        args.run_mode = input('Choose scenario: gridsearch | train | query :> ').strip()

    tmp_dir = args.tmp_dir
    data_dir = args.data_dir
    run_mode = args.run_mode

    arch_path = os.path.join(tmp_dir, 'nn_seq2seq_interpreter.arch')
    config_path = os.path.join(tmp_dir, 'nn_seq2seq_interpreter.config')

    if run_mode == 'gridsearch':
        weights_path = os.path.join(tmp_dir, 'gridsearch.nn_seq2seq_interpreter.weights.tmp')
    else:
        weights_path = os.path.join(tmp_dir, 'nn_seq2seq_interpreter.weights')

    batch_size = 32

    if run_mode == 'gridsearch':
        best_score = 0
        best_params_path = os.path.join(tmp_dir, 'nn_seq2seq_interpreter.best_params.json')

        for spm_items in [5000, 10000]:
            for token_dim in [50, 80]:
                for hidden_dim in [150, 200]:
                    for is_monotonic in [False]:
                        params = dict()
                        params['spm_items'] = spm_items
                        params['token_dim'] = token_dim
                        params['hidden_dim'] = hidden_dim
                        params['is_monotonic'] = is_monotonic

                        computed_params = dict()
                        bpe_model_name = train_bpe_model(params)
                        computed_params['bpe_model_name'] = bpe_model_name

                        bpe_model = load_bpe_model(bpe_model_name)
                        samples = load_samples(bpe_model, computed_params, max_samples=20000)

                        index2token = dict((i, t) for t, i in computed_params['token2index'].items())

                        kf = KFold(n_splits=3)
                        scores = []
                        for ifold, (train_index, test_index) in enumerate(kf.split(samples)):
                            train_samples = [samples[i] for i in train_index]
                            test_samples = [samples[i] for i in test_index]

                            Xs_train, y_train = vectorize_samples(train_samples, params, computed_params)
                            y_train = np.expand_dims(y_train, -1)

                            model = create_model(params, computed_params)

                            callbacks = []
                            model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
                            callbacks.append(model_checkpoint)

                            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
                            callbacks.append(early_stopping)

                            model.fit(Xs_train, y_train, validation_split=0.1, epochs=100, verbose=2, batch_size=batch_size, callbacks=callbacks)
                            model.load_weights(weights_path)

                            # получим метрику качества этой модели
                            nb_good = 0
                            nb_total = 0
                            test_batch_size = 100

                            for isample in range(0, len(test_samples), test_batch_size):
                                batch_samples = test_samples[isample:isample+batch_size]
                                Xs_batch, y_batch = vectorize_samples(batch_samples, params, computed_params)
                                y_pred = model.predict(Xs_batch, batch_size=batch_size, verbose=0)
                                for sample, sample_y_pred, y_true in zip(test_samples, y_pred, y_batch):
                                    sample_y_pred = np.argmax(sample_y_pred, axis=-1)
                                    nb_total += 1
                                    if np.array_equal(sample_y_pred, y_true):
                                        nb_good += 1

                            fold_score = nb_good/float(nb_total)
                            scores.append(fold_score)

                        score = np.mean(scores)
                        print('Cross-val score={}'.format(score))
                        if score > best_score:
                            best_score = score
                            params_str = get_params_str(params)
                            print('!!! NEW BEST score={} params='.format(best_score, ))
                            with open(best_params_path, 'w') as f:
                                json.dump(params, f, indent=4)
                        else:
                            print('No improvement over best_score={}'.format(best_score))

    if run_mode == 'train':
        params = dict()

        params['spm_items'] = 16000
        params['token_dim'] = 50
        params['hidden_dim'] = 200
        params['is_monotonic'] = False
        params['encode_segments'] = False
        params['crf'] = False

        computed_params = dict()
        computed_params['arch_path'] = arch_path
        computed_params['weights_path'] = weights_path

        bpe_model_name = train_bpe_model(params)
        computed_params['bpe_model_name'] = bpe_model_name

        bpe_model = load_bpe_model(bpe_model_name)

        samples = load_samples(bpe_model, computed_params, max_samples=1000000)
        train_samples, test_samples = sklearn.model_selection.train_test_split(samples, test_size=0.1)

        with open(config_path, 'w') as f:
            config = dict(computed_params)
            config['encode_segments'] = params['encode_segments']
            json.dump(config, f, indent=4)

        print('Vectorization of {} samples for training'.format(len(train_samples)))
        Xs, y = vectorize_samples(train_samples, params, computed_params)
        y = np.expand_dims(y, -1)
        print('X1.shape={}'.format(Xs[0].shape))
        print('y.shape={}'.format(y.shape))

        model = create_model(params, computed_params)

        with open(arch_path, 'w') as f:
            f.write(model.to_json())

        callbacks = []
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        #callbacks.append(model_checkpoint)

        #early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        #callbacks.append(early_stopping)

        viz = VizualizeCallback(model, test_samples, params, computed_params, weights_path, patience=15)
        callbacks.append(viz)

        model.fit(x=Xs, y=y,
                  validation_split=0.0, validation_data=None,
                  epochs=500, verbose=2, batch_size=batch_size, shuffle=True,
                  callbacks=callbacks)

    if run_mode == 'report':
        # Финальная оценка "на глазок" по всем сэмплам (будет слишком оптимистичная оценка, конечно).

        # Загружаем конфиг натренированной модели
        with open(config_path, 'r') as f:
            computed_params = json.load(f)
            arch_file = os.path.join(tmp_dir, os.path.basename(computed_params['arch_path']))
            weights_file = os.path.join(tmp_dir, os.path.basename(computed_params['weights_path']))
            bpe_model_name = computed_params['bpe_model_name']

        with open(arch_file, 'r') as f:
            model = model_from_json(f.read(), {'AttentionDecoder': AttentionDecoder})

        model.load_weights(weights_file)

        bpe_model = load_bpe_model(bpe_model_name)

        # Все сэмплы
        dummy = dict()
        samples = load_samples(bpe_model, dummy, max_samples=1000000)

        print('Final assessment on {} samples'.format(len(samples)))
        nb_good = 0
        nb_total = 0
        batch_size = 200
        index2token = dict((i, t) for t, i in computed_params['token2index'].items())

        with io.open(os.path.join(tmp_dir, 'train_nn_seq2seq_interpreter.report.txt'), 'w', encoding='utf-8') as wrt, \
             io.open(os.path.join(tmp_dir, 'train_nn_seq2seq_interpreter.error_samples.txt'), 'w', encoding='utf-8') as wrt_err:
            for isample in range(0, len(samples), batch_size):
                batch_samples = samples[isample:isample+batch_size]
                Xs_batch, y_batch = vectorize_samples(batch_samples, computed_params, computed_params)
                y_pred = model.predict(Xs_batch, batch_size=batch_size, verbose=0)
                for sample, sample_y_pred, y_true in zip(batch_samples, y_pred, y_batch):
                    sample_y_pred = np.argmax(sample_y_pred, axis=-1)

                    tokens = [index2token[itok] for itok in sample_y_pred]
                    pred_right = ''.join(tokens).replace('▁', ' ').strip()

                    wrt.write('\n\nContext:          {}\n'.format(sample.left_str))
                    wrt.write('True answer:      {}\n'.format(sample.right_str))
                    wrt.write('Predicted answer: {}\n'.format(pred_right))

                    # Точность per instance
                    nb_total += 1
                    if pred_right == sample.right_str:  #np.array_equal(sample_y_pred, y_true):
                        nb_good += 1
                    else:
                        # Выведем в текстовый ответ описание ошибки
                        context = '\n'.join([s.strip() for s in sample.left_str.split('|')])
                        context = context.replace(' ?', '?').replace(' .', '.').replace(' ,', ',')
                        wrt_err.write('{} | {}\n\n\n'.format(context, sample.right_str.replace(' .', '.')))

        print('Accuracy per instance={}'.format(nb_good/float(nb_total)))

    if run_mode == 'query':
        # Ввод контекста в консоли, выдача предсказанной интерпретации

        # Загружаем конфиг натренированной модели
        with open(config_path, 'r') as f:
            computed_params = json.load(f)
            arch_file = os.path.join(tmp_dir, os.path.basename(computed_params['arch_path']))
            weights_file = os.path.join(tmp_dir, os.path.basename(computed_params['weights_path']))
            bpe_model_name = computed_params['bpe_model_name']

        with open(arch_file, 'r') as f:
            model = model_from_json(f.read(), {'AttentionDecoder': AttentionDecoder})

        model.load_weights(weights_file)

        bpe_model = load_bpe_model(bpe_model_name)

        index2token = dict((i, t) for t, i in computed_params['token2index'].items())

        while True:
            context = []
            print('Enter context phrases, empty line to run the model:')
            while True:
                s = input('{}:> '.format(len(context)+1)).strip()
                if s:
                    context.append(s)
                else:
                    break

            context = ' | '.join(s.lower() for s in context)
            print('Context: {}'.format(context))

            sample = create_prediction_sample(bpe_model, context)
            Xs, _ = vectorize_samples([sample], computed_params, computed_params)
            y_pred = model.predict(Xs, batch_size=batch_size, verbose=0)
            y_pred = np.argmax(y_pred[0], axis=-1)
            tokens = [index2token[itok] for itok in y_pred]
            pred_right = ''.join(tokens).replace('▁', ' ').strip()
            print('Output: {}\n\n'.format(pred_right))

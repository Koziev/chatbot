"""
Тренировка модели для генерации ответа в PQA (при заданной предпосылке и вопросе)
с помощью seq2seq и attention
"""

import io
import os
import itertools
import random
import json
import argparse


import numpy as np
import sklearn.model_selection
from sklearn.model_selection import KFold

import sentencepiece as spm
import terminaltables

import keras
import keras.callbacks
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json

# https://github.com/asmekal/keras-monotonic-attention
from ruchatbot.layers.attention_decoder import AttentionDecoder



def get_params_str(model_params):
    return ' '.join('{}={}'.format(k, v) for (k, v) in model_params.items())


def dress_context_line(s):
    if s[-1] in '.?!':
        return s
    else:
        return s + ' .'


def train_bpe_model(params):
    spm_items = params['spm_items']

    # Готовим корпус для обучения SentencePiece
    sentencepiece_corpus = os.path.join(tmp_dir, 'sentencepiece_corpus.txt')

    nb_samples = 0
    #max_nb_samples = 10000000  # макс. кол-во предложений для обучения SentencePiece
    with io.open(sentencepiece_corpus, 'w', encoding='utf-8') as wrt:
        with io.open(os.path.join(data_dir, 'pqa_all.dat'), 'r', encoding='utf-8') as rdr:
            lines = []
            for line in rdr:
                line = line.strip()
                if line:
                    lines.append(line)
                else:
                    if lines:
                        left_data = [dress_context_line(s) for s in lines[:-2]] + [lines[-2] + ' ?']
                        left_data = ' '.join(left_data)
                        right_data = lines[-1]

                        wrt.write('{}\n'.format(left_data))
                        wrt.write('{}\n'.format(right_data))
                        nb_samples += 1
                        #if nb_samples >= max_nb_samples:
                        #    break

                    lines = []

    spm_name = 'nn_seq2seq_pqa_generator.sentencepiece'  #.format(spm_items)

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
        self.right_str = None
        self.right_tokens = None


def load_samples(bpe_model, computed_params, max_samples):
    all_tokens = set()
    samples_yes = []
    samples_others = []
    with io.open(os.path.join(data_dir, 'pqa_all.dat'), 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if line:
                lines.append(line)
            else:
                if lines:
                    left_data = [dress_context_line(s) for s in lines[:-2]] + [lines[-2] + ' ?']
                    sample = Sample()
                    sample.left_str = ' '.join(left_data)
                    sample.left_tokens = bpe_model.EncodeAsPieces(sample.left_str)
                    sample.right_str = lines[-1]
                    sample.right_tokens = bpe_model.EncodeAsPieces(sample.right_str)

                    if sample.right_str == 'да':
                        samples_yes.append(sample)
                    else:
                        samples_others.append(sample)

                    all_tokens.update(sample.left_tokens)
                    all_tokens.update(sample.right_tokens)

                lines = []

    # Ограничим количество сэмплов с ответом 'да'
    #nyes = len(samples_others) // 20
    #if len(samples_yes) > nyes:
    #    samples_yes = sorted(samples_yes, key=lambda z: random.random())[:nyes]
    print('samples_yes.count={}'.format(len(samples_yes)))

    # Объединяем сэмплы
    samples = samples_others + samples_yes
    samples = sorted(samples, key=lambda z: random.random())
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

    return samples


def vectorize_samples(samples, computed_params):
    nb_samples = len(samples)

    token2index = computed_params['token2index']
    max_left_len = computed_params['max_left_len']
    max_right_len = computed_params['max_right_len']
    seq_len = max(max_left_len, max_right_len)

    X = np.zeros((nb_samples, seq_len), dtype=np.int32)
    y = np.zeros((nb_samples, seq_len), dtype=np.int32)
    for isample, sample in enumerate(samples):
        for itoken, token in enumerate(sample.left_tokens[:seq_len]):
            X[isample, itoken] = token2index[token]

        for itoken, token in enumerate(sample.right_tokens[:seq_len]):
            y[isample, itoken] = token2index[token]

    return X, y


def create_model(params, computed_params):
    model = Sequential()

    seq_len = max(computed_params['max_left_len'], computed_params['max_right_len'])
    hidden_dim = params['hidden_dim']

    model.add(Embedding(input_dim=params['spm_items'], output_dim=params['token_dim'], input_length=seq_len))
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))
    f = params['is_monotonic']
    model.add(AttentionDecoder(units=hidden_dim, alphabet_size=params['spm_items'],
                               embedding_dim=params['token_dim'],
                               is_monotonic=f, normalize_energy=f))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')  # , metrics=['acc'])
    model.summary()
    return model


class VizualizeCallback(keras.callbacks.Callback):
    """
    После каждой эпохи обучения делаем сэмплинг образцов из текущей модели,
    чтобы видеть общее качество.
    """

    def __init__(self, model, viz_samples, computed_params):
        self.model = model
        self.computed_params = computed_params
        self.viz_samples = viz_samples

    def on_epoch_end(self, batch, logs={}):
        samples2 = sorted(self.viz_samples, key=lambda _: random.random())[:5]
        X, y = vectorize_samples(samples2, self.computed_params)
        y_pred = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred, axis=-1)

        index2token = dict((i, t) for t, i in computed_params['token2index'].items())

        table = ['context true_output predicted_output'.split()]
        for sample, y_pred_sample in zip(samples2, y_pred):
            # Декодируем список индексов предсказанных токенов
            tokens = [index2token[itok] for itok in y_pred_sample]
            pred_right = ''.join(tokens).replace('▁', ' ').strip()
            table.append((sample.left_str, sample.right_str, pred_right))

        table = terminaltables.AsciiTable(table)
        print(table.table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Answer generation model trainer')
    parser.add_argument('--run_mode', choices='gridsearch train query'.split(), default='report')
    parser.add_argument('--tmp_dir', default='../../tmp')
    parser.add_argument('--data_dir', default='../../data')
    args = parser.parse_args()

    tmp_dir = args.tmp_dir
    data_dir = args.data_dir
    run_mode = args.run_mode

    arch_path = os.path.join(tmp_dir, 'nn_seq2seq_pqa_generator.arch')
    config_path = os.path.join(tmp_dir, 'nn_seq2seq_pqa_generator.config')

    if run_mode == 'gridsearch':
        weights_path = os.path.join(tmp_dir, 'gridsearch.nn_seq2seq_pqa_generator.weights.tmp')
    else:
        weights_path = os.path.join(tmp_dir, 'nn_seq2seq_pqa_generator.weights')

    batch_size = 100

    if run_mode == 'gridsearch':
        best_score = 0
        best_params_path = os.path.join(tmp_dir, 'nn_seq2seq_pqa_generator.best_params.json')

        for spm_items in [20000, 30000]:
            for token_dim in [50, 80]:
                for hidden_dim in [150, 200]:
                    for is_monotonic in [False, True]:
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

                            X_train, y_train = vectorize_samples(train_samples, computed_params)
                            y_train = np.expand_dims(y_train, -1)

                            model = create_model(params, computed_params)

                            callbacks = []
                            model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
                            callbacks.append(model_checkpoint)

                            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
                            callbacks.append(early_stopping)

                            model.fit(X_train, y_train, validation_split=0.1, epochs=100, verbose=2, batch_size=batch_size, callbacks=callbacks)
                            model.load_weights(weights_path)

                            # получим метрику качества этой модели
                            nb_good = 0
                            nb_total = 0
                            test_batch_size = 100

                            for isample in range(0, len(test_samples), test_batch_size):
                                batch_samples = test_samples[isample:isample+batch_size]
                                X_batch, y_batch = vectorize_samples(batch_samples, computed_params)
                                y_pred = model.predict(X_batch, batch_size=batch_size, verbose=0)
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

        params['spm_items'] = 20000
        params['token_dim'] = 50
        params['hidden_dim'] = 150
        params['is_monotonic'] = False

        computed_params = dict()
        computed_params['arch_path'] = arch_path
        computed_params['weights_path'] = weights_path

        bpe_model_name = train_bpe_model(params)
        computed_params['bpe_model_name'] = bpe_model_name

        bpe_model = load_bpe_model(bpe_model_name)

        samples = load_samples(bpe_model, computed_params, max_samples=1000000)

        with open(config_path, 'w') as f:
            json.dump(computed_params, f, indent=4)

        # Для визуализации отберем сэмплы, в которых ответ не "да"
        viz_samples1 = filter(lambda z: z.right_str != 'да', samples)
        viz_samples2 = filter(lambda z: z.right_str == 'да', samples)
        viz_samples = list(viz_samples1)[:80] + list(viz_samples2)[:5]

        print('Vectorization of {} samples'.format(len(samples)))
        X, y = vectorize_samples(samples, computed_params)
        y = np.expand_dims(y, -1)
        print('X.shape={}'.format(X.shape))
        print('y.shape={}'.format(y.shape))

        model = create_model(params, computed_params)

        with open(arch_path, 'w') as f:
            f.write(model.to_json())

        callbacks = []
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks.append(model_checkpoint)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        callbacks.append(early_stopping)

        viz = VizualizeCallback(model, viz_samples, computed_params)
        callbacks.append(viz)

        model.fit(X, y, validation_split=0.1, epochs=100, verbose=2, batch_size=batch_size, callbacks=callbacks)

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

        with io.open(os.path.join(tmp_dir, 'train_nn_seq2seq_pqa_generator.report.txt'), 'w', encoding='utf-8') as wrt,\
             io.open(os.path.join(tmp_dir, 'train_nn_seq2seq_pqa_generator.errors.txt'), 'w', encoding='utf-8') as wrt2:
            for isample in range(0, len(samples), batch_size):
                batch_samples = samples[isample:isample+batch_size]
                X_batch, y_batch = vectorize_samples(batch_samples, computed_params)
                y_pred = model.predict(X_batch, batch_size=batch_size, verbose=0)
                for sample, sample_y_pred, y_true in zip(samples, y_pred, y_batch):
                    sample_y_pred = np.argmax(sample_y_pred, axis=-1)

                    tokens = [index2token[itok] for itok in sample_y_pred]
                    pred_right = ''.join(tokens).replace('▁', ' ').strip()
                    wrt.write('\n\nContext:          {}\n'.format(sample.left_str))
                    wrt.write('True answer:      {}\n'.format(sample.right_str))
                    wrt.write('Predicted answer: {}\n'.format(pred_right))

                    nb_total += 1
                    if np.array_equal(sample_y_pred, y_true):
                        nb_good += 1
                    else:
                        sx = [s.strip() for s in sample.left_str.split('|')]
                        premises = sx[:-1]
                        question = sx[-1]
                        wrt2.write('\n\n')
                        for premise in premises:
                            wrt2.write('T: {}\n'.format(premise))
                        wrt2.write('Q: {}\n'.format(question))
                        wrt2.write('A: {}\n'.format(sample.right_str))

        print('Dirty accuracy={}'.format(nb_good/float(nb_total)))

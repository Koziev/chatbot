# -*- coding: utf-8 -*-
"""
Тренировка модели интерпретации реплик в чатботе.

Для вопросно-ответной системы https://github.com/Koziev/chatbot.

Используется специализированный датасет с примерами интерпретаций, а
также автоматически сгенерированные датасеты.
"""

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os
import gensim
import io
import numpy as np
import tqdm
import argparse
import random
import logging

from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.callbacks
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.layers.core import RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.models import model_from_json
from keras.layers.wrappers import TimeDistributed

from sklearn.model_selection import train_test_split
import sklearn.metrics

from utils.tokenizer import Tokenizer
from utils.lemmatizer import Lemmatizer
import utils.console_helpers
import utils.logging_helpers


PAD_WORD = u''
padding = 'left'
nb_filters = 128
max_kernel_size = 2



class Sample:
    def __init__(self, phrases, phrase_words, phrase_lemmas, result_phrase, result_lemmas):
        self.phrases = phrases[:]
        self.phrase_words = phrase_words[:]
        self.phrase_lemmas = phrase_lemmas[:]
        self.result_phrase = result_phrase
        self.result_lemmas = result_lemmas


def get_first_part(s):
    if u'|' in s:
        return s.split(u'|')[0]
    else:
        return s


def load_samples(input_path, tokenizer, lemmatizer):
    logging.info(u'Loading samples from {}'.format(input_path))
    samples = []

    with io.open(input_path, 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()

            if line.startswith('#'):  # пропускаем комментарии
                continue

            if len(line) == 0:
                if len(lines) > 0:
                    for nline in range(1, len(lines)+1):
                        sample_lines = lines[:nline]

                        # Фразы предшествующего контекста
                        phrases = [get_first_part(s) for s in sample_lines[:-1]]

                        # Последняя фраза с результатом подстановок и интерпретации
                        last_line = sample_lines[-1]

                        if '|' in last_line:
                            fx = last_line.split('|')
                            phrases.append(fx[0])  # добавим левую часть последней пары к контексту
                            phrase_words = [tokenizer.tokenize(f) for f in phrases]
                            phrase_lemmas = [lemmatizer.tokenize(f) for f in phrases]

                            result_phrase = fx[1]  # правая часть последней пары становится результатом интерпретации
                            result_lemmas = lemmatizer.tokenize(result_phrase)

                            sample = Sample(phrases, phrase_words, phrase_lemmas, result_phrase, result_lemmas)
                            samples.append(sample)

                    lines = []
            else:
                lines.append(line)

    logging.info(u'{} samples loaded from {}'.format(len(samples), input_path))
    return samples


def ngrams(s, n):
    return set(itertools.izip(*[s[i:] for i in range(n)]))


def pad_wordseq(words, n):
    """Слева добавляем пустые слова"""
    return list(itertools.chain(itertools.repeat(PAD_WORD, n - len(words)), words))


def rpad_wordseq(words, n):
    """Справа добавляем пустые слова"""
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def vectorize_words(words, M, irow, word2vec):
    for iword, word in enumerate(words):
        if word != PAD_WORD:
            if word in word2vec:
                M[irow, iword, :] = word2vec[word]
            else:
                logging.error(u'Word \"{}\" is missing in word2vec; phrase={}'.format(word, u' '.join(words).strip()))
                raise RuntimeError()


def generate_rows(nb_inputs, output_dims, max_outseq_len, samples, lemma2id, batch_size, mode):
    batch_index = 0
    batch_count = 0

    Xn_batch = []
    Xlemmas_batch = []
    for _ in range(nb_inputs):
        x = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
        Xn_batch.append(x)
        x = np.zeros((batch_size, output_dims), dtype=np.float32)
        Xlemmas_batch.append(x)

    inputs = {}
    for iphrase in range(nb_inputs):
        inputs['input{}'.format(iphrase)] = Xn_batch[iphrase]
        inputs['lemmas{}'.format(iphrase)] = Xlemmas_batch[iphrase]

    y_batch = np.zeros((batch_size, max_outseq_len, output_dims), dtype=np.float32)

    while True:
        for irow, sample in enumerate(samples):
            for iphrase, words in enumerate(sample.phrase_words):
                words = pad_wordseq(words, max_inputseq_len)
                vectorize_words(words, Xn_batch[iphrase], batch_index, word2vec)

            for iphrase, lemmas in enumerate(sample.phrase_lemmas):
                for lemma in lemmas:
                    Xlemmas_batch[iphrase][batch_index, lemma2id[lemma]] = 1

            for ilemma, lemma in enumerate(sample.result_lemmas):
                y_batch[batch_index, ilemma, lemma2id[lemma]] = 1

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield (inputs, {'output': y_batch})
                else:
                    yield inputs

                # очищаем матрицы порции для новой порции
                for x in Xn_batch:
                    x.fill(0)

                for x in Xlemmas_batch:
                    x.fill(0)

                y_batch.fill(0)
                batch_index = 0



class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class VisualizeCallback(keras.callbacks.Callback):
    def __init__(self, max_nb_inputs, val_samples, model, weights_path, lemma2id, max_outseq_len):
        self.epoch = 0
        self.max_nb_inputs = max_nb_inputs
        self.max_outseq_len = max_outseq_len
        self.val_samples = val_samples
        self.model = model
        self.weights_path = weights_path
        self.lemma2id = lemma2id
        self.id2lemma = dict((id, lemma) for (lemma, id) in lemma2id.items())
        self.nb_outputs = len(lemma2id)
        self.best_acc = 0
        self.stop_epoch = 0
        self.early_stopping = 20
        self.wait_epoch = 0
        self.val_acc_history = []  # для сохранения кривой обучения

    def decode_ystr(self, y):
        s = []
        for lemma_v in y:
            lemma_index = np.argmax(lemma_v)
            lemma = self.id2lemma[lemma_index]
            if lemma != PAD_WORD:
                s.append(lemma)

        return u' '.join(s)

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        nval = len(self.val_samples)

        nb_errors = 0

        # Счетчик напечатанных строк, сгенерированных моделью
        nb_shown = 0

        nb_steps = nval//batch_size

        print('')
        acc = 0.0
        acc_denom = 0.0
        for step, batch in enumerate(generate_rows(self.max_nb_inputs, self.nb_outputs, self.max_outseq_len, self.val_samples, self.lemma2id, batch_size, 1)):
            if step == nb_steps:
                break

            y_batch = batch[1]['output']
            y_pred = model.predict_on_batch(batch[0])

            for iy in range(len(y_pred)):
                predicted_lemmas = self.decode_ystr(y_pred[iy])
                target_lemmas = self.decode_ystr(y_batch[iy])

                is_good = target_lemmas == predicted_lemmas
                acc += is_good
                acc_denom += 1.0

                if nb_shown < 10:
                    print(
                        colors.ok + '☑ ' + colors.close if is_good else colors.fail + '☒ ' + colors.close,
                        end='')

                    print(u'true={} model={}'.format(target_lemmas, predicted_lemmas))
                    nb_shown += 1

        acc = acc / acc_denom
        self.val_acc_history.append(acc)
        if acc > self.best_acc:
            utils.console_helpers.print_green_line('New best instance accuracy={}'.format(acc))
            self.wait_epoch = 0
            print(u'Saving model to {}'.format(self.weights_path))
            self.model.save_weights(self.weights_path)
            self.best_acc = acc
        else:
            self.wait_epoch += 1
            print('\nInstance accuracy={} did not improve ({} epochs since last improvement)\n'.format(acc, self.wait_epoch))
            if self.wait_epoch >= self.early_stopping:
                print('Training stopped after {} epochs without improvement'.format(self.wait_epoch))
                print('Best instance accuracy={}'.format(self.best_acc))
                self.model.stop_training = True
                self.stop_epoch = self.epoch

    def save_learning_curve(self, path):
        with open(path, 'w') as wrt:
            wrt.write('epoch\tacc\n')
            for i, acc in enumerate(self.val_acc_history):
                wrt.write('{}\t{}\n'.format(i+1, acc))


# -------------------------------------------------------------------

# классы для алгоритма декодера
class InterpreterTrellisNode:
    def __init__(self, lemma, word):
        self.lemma = lemma
        self.word = word
        self.is_start = False
        self.is_end = False
        self.best_p = 1.0
        self.best_prev = None

    @classmethod
    def create_start(cls):
        s = InterpreterTrellisNode(u'<S>', u'<S>')
        s.is_start = True
        return s

    @classmethod
    def create_end(cls):
        s = InterpreterTrellisNode(u'<E>', u'<E>')
        s.is_end = True
        return s


class TrellisColumn:
    def __init__(self):
        self.cells = []

    def add_cell(self, cell):
        self.cells.append(cell)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural interpreter model')
    parser.add_argument('--run_mode', type=str, default='train', choices='train query'.split(), help='what to do: train | query')
    parser.add_argument('--arch', type=str, default='lstm(cnn)', help='neural model architecture: lstm | lstm(cnn)')
    parser.add_argument('--batch_size', type=int, default=250, help='batch size for neural model training')
    parser.add_argument('--tmp', type=str, default='../tmp', help='folder to store results')
    parser.add_argument('--wordchar2vector', type=str, default='../data/wordchar2vector.dat', help='path to wordchar2vector model dataset')
    parser.add_argument('--word2vector', type=str, default='~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin', help='path to word2vector model file')
    parser.add_argument('--data_dir', type=str, default='../data', help='folder containing some evaluation datasets')

    args = parser.parse_args()

    data_folder = args.data_dir
    tmp_folder = args.tmp

    wordchar2vector_path = args.wordchar2vector
    word2vector_path = os.path.expanduser(args.word2vector)
    batch_size = args.batch_size
    net_arch = args.arch
    run_mode = args.run_mode

    utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'nn_interpreter.log'))

    # В этих файлах будем сохранять натренированную сетку
    config_path = os.path.join(tmp_folder, 'nn_interpreter.config')
    arch_filepath = os.path.join(tmp_folder, 'nn_interpreter.arch')
    weights_path = os.path.join(tmp_folder, 'nn_interpreter.weights')


    if run_mode == 'train':
        logging.info('Start with run_mode==train')

        logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path))
        wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
        wc2v_dims = len(wc2v.syn0[0])
        logging.info('wc2v_dims={0}'.format(wc2v_dims))

        # Загружаем датасеты, содержащие сэмплы ПРЕДПОСЫЛКИ-ВОПРОС-ОТВЕТ

        tokenizer = Tokenizer()
        tokenizer.load()

        lemmatizer = Lemmatizer()

        good_lens = (1, 2, 3)

        # это вручную созданные сэмплы.
        samples1 = load_samples('../data/interpretation.txt', tokenizer, lemmatizer)
        samples1 = list(filter(lambda s: len(s.phrases) in good_lens, samples1))

        # остальные сэмплы получаются при автогенерации
        samples_auto4 = load_samples('../data/interpretation_auto_4.txt', tokenizer, lemmatizer)
        samples_auto5 = load_samples('../data/interpretation_auto_5.txt', tokenizer, lemmatizer)
        samples_auto = list(itertools.chain(samples_auto4, samples_auto5))
        samples_auto = np.random.permutation(samples_auto)[:len(samples1)*500]

        samples = list(itertools.chain(samples1, samples_auto))

        # для отладки - оставим только сэмплы с 2 строками контекста
        samples = list(filter(lambda s: len(s.phrases) in good_lens, samples))

        logging.info('samples.count={}'.format(len(samples)))

        max_nb_inputs = max(len(sample.phrases) for sample in samples)
        max_inputseq_len = 0
        max_outseq_len = 0
        all_words = set()
        all_lemmas = set([PAD_WORD])
        for sample in samples:
            for words in sample.phrase_words:
                all_words.update(words)
                max_inputseq_len = max(max_inputseq_len, len(words))
            for lemmas in sample.phrase_lemmas:
                all_lemmas.update(lemmas)
            all_lemmas.update(sample.result_lemmas)
            max_outseq_len = max(max_outseq_len, len(sample.result_lemmas))

        logging.info('max_inputseq_len={}'.format(max_inputseq_len))
        logging.info('max_outseq_len={}'.format(max_outseq_len))
        logging.info('max_nb_inputs={}'.format(max_nb_inputs))

        nb_lemmas = len(all_lemmas)
        lemma2id = dict((lemma, i) for (i, lemma) in enumerate(itertools.chain([PAD_WORD], filter(lambda l: l != PAD_WORD, all_lemmas))))
        logging.info('nb_lemmas={}'.format(nb_lemmas))

        for word in wc2v.vocab:
            all_words.add(word)

        word2id = dict([(c, i) for i, c in enumerate(itertools.chain([PAD_WORD], filter(lambda z: z != PAD_WORD,all_words)))])
        nb_words = len(all_words)
        logging.info('nb_words={}'.format(nb_words))

        for sample in samples:
            sample.result_lemmas = rpad_wordseq(sample.result_lemmas, max_outseq_len)

        # --------------------------------------------------------------------------

        logging.info('Loading the w2v model {}'.format(word2vector_path))
        w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
        w2v_dims = len(w2v.syn0[0])
        logging.info('w2v_dims={0}'.format(w2v_dims))

        word_dims = w2v_dims+wc2v_dims

        word2vec = dict()
        for word in wc2v.vocab:
            v = np.zeros(word_dims)
            v[w2v_dims:] = wc2v[word]
            if word in w2v:
                v[:w2v_dims] = w2v[word]

            word2vec[word] = v

        del w2v
        #del wc2v
        gc.collect()
        # --------------------------------------------------------------------------------

        # сохраним конфиг модели, чтобы ее использовать в чат-боте
        model_config = {
                        'engine': 'nn',
                        'max_inputseq_len': max_inputseq_len,
                        'max_outseq_len': max_outseq_len,
                        'w2v_path': word2vector_path,
                        'wordchar2vector_path': wordchar2vector_path,
                        'PAD_WORD': PAD_WORD,
                        'padding': padding,
                        'model_folder': tmp_folder,
                        'word_dims': word_dims,
                        'lemma2id': lemma2id,
                        'max_nb_inputs': max_nb_inputs,
                        'arch_filepath': arch_filepath,
                        'weights_filepath': weights_path
                       }

        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=4)

        logging.info('Constructing neural net: {}...'.format(net_arch))

        rnn_size = word_dims

        inputs = []
        for iphrase in range(max_nb_inputs):
            input_phrase = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input{}'.format(iphrase))
            inputs.append(input_phrase)

        lemmas_inputs = []
        for iphrase in range(max_nb_inputs):
            input_lemmas = Input(shape=(nb_lemmas,), dtype='float32', name='lemmas{}'.format(iphrase))
            lemmas_inputs.append(input_lemmas)

        layers = []
        layers.extend(lemmas_inputs)

        encoder_size = 0

        if net_arch == 'lstm':
            # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
            # предложения. Этот слой общий для всех входных предложений.
            shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size,
                                                            input_shape=(max_inputseq_len, word_dims),
                                                            return_sequences=False))

            for input in inputs:
                encoder_rnn = shared_words_rnn(input)
                layers.append(encoder_rnn)
                encoder_size += rnn_size*2
        elif net_arch == 'lstm(cnn)':
            for kernel_size in range(1, max_kernel_size+1):
                # сначала идут сверточные слои, образующие детекторы словосочетаний
                # и синтаксических конструкций
                conv = Conv1D(filters=nb_filters,
                              kernel_size=kernel_size,
                              padding='valid',
                              activation='relu',
                              strides=1,
                              name='shared_conv_{}'.format(kernel_size))

                lstm = recurrent.LSTM(rnn_size, return_sequences=False)

                for input in inputs:
                    conv_layer1 = conv(input)
                    conv_layer1 = keras.layers.MaxPooling1D(pool_size=kernel_size,
                                                            strides=None,
                                                            padding='valid')(conv_layer1)
                    conv_layer1 = lstm(conv_layer1)
                    layers.append(conv_layer1)
                    encoder_size += rnn_size
        else:
            raise NotImplementedError('net_arch={} is not implemented'.format(net_arch))

        output_dims = nb_lemmas

        encoder_merged = keras.layers.concatenate(inputs=list(layers))

        decoder = encoder_merged
        rnn_size = 100
        decoder = Dense(units=rnn_size, activation='relu')(decoder)
        #    decoder = Dense(units=encoder_size, activation='relu')(decoder)
        #    decoder = Dense(units=encoder_size, activation='relu')(decoder)
        decoder = RepeatVector(max_outseq_len)(decoder)
        decoder = recurrent.LSTM(rnn_size, return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(nb_lemmas, activation='softmax'), name='output')(decoder)

        model = Model(inputs=list(itertools.chain(inputs, lemmas_inputs)), outputs=decoder)
        model.compile(loss='categorical_crossentropy', optimizer='nadam')
        model.summary()

        with open(arch_filepath, 'w') as f:
            f.write(model.to_json())

        SEED = 123456
        TEST_SHARE = 0.2
        train_samples, val_samples = train_test_split(samples, test_size=TEST_SHARE, random_state=SEED)

        nb_train_patterns = len(train_samples)
        nb_valid_patterns = len(val_samples)

        logging.info('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

        monitor_metric = 'val_loss'

        #model_checkpoint = ModelCheckpoint(weights_path,
        #                                   monitor=monitor_metric,
        #                                   verbose=1,
        #                                   save_best_only=True,
        #                                   mode='auto')
        #early_stopping = EarlyStopping(monitor=monitor_metric, patience=20, verbose=1, mode='auto')

        viz = VisualizeCallback(max_nb_inputs, val_samples, model, weights_path, lemma2id, max_outseq_len)

        #callbacks = [viz, model_checkpoint, early_stopping]
        callbacks = [viz]

        hist = model.fit_generator(generator=generate_rows(max_nb_inputs, nb_lemmas, max_outseq_len, train_samples, lemma2id, batch_size, 1),
                                   steps_per_epoch=nb_train_patterns//batch_size,
                                   epochs=200,  # 1000
                                   verbose=1,
                                   callbacks=callbacks,
                                   validation_data=generate_rows(max_nb_inputs, nb_lemmas, max_outseq_len, val_samples, lemma2id, batch_size, 1),
                                   validation_steps=nb_valid_patterns//batch_size)

        logging.info('Best accuracy per instance={}'.format(viz.best_acc))

        # Прогоним через модель все сэмплы, сохраним декодированные результаты в текстовом файле
        # для визуального анализа плохих и хороших сэмплов.
        logging.info('Final validation of model...')
        model.load_weights(weights_path)

        true_results = []
        predicted_results = []
        samples_viz = samples1  # для этих сэмплов будем делать визуализацию результатов
        nb_steps = len(samples_viz) // batch_size
        for istep, xy in tqdm.tqdm(enumerate(generate_rows(max_nb_inputs, nb_lemmas, max_outseq_len, samples_viz, lemma2id, batch_size, 1)),
                                   total=nb_steps,
                                   desc='Running model'):
            x = xy[0]
            y_batch = xy[1]['output']
            y_pred = model.predict(x=x, verbose=0)
            for iy in range(len(y_pred)):
                predicted_lemmas = viz.decode_ystr(y_pred[iy])
                target_lemmas = viz.decode_ystr(y_batch[iy])

                true_results.append(target_lemmas)
                predicted_results.append(predicted_lemmas)

            if istep >= nb_steps:
                break

        logging.info('Writing {} samples with predictions...'.format(len(predicted_results)))
        with io.open(os.path.join(tmp_folder, 'nn_interpreter.validation.txt'), 'w', encoding='utf-8') as wrt:
            for isample, sample in enumerate(samples_viz[:len(predicted_results)]):
                correctness = 'OK'
                if true_results[isample] != predicted_results[isample]:
                    correctness = 'ERROR'
                wrt.write(u'isample={} {}\n'.format(isample, correctness))
                for iphrase, phrase in enumerate(sample.phrases):
                    wrt.write(u'phrase[{}]={}\n'.format(iphrase, phrase))
                wrt.write(u'true result     ={}\n'.format(u' '.join(sample.result_lemmas)))
                wrt.write(u'predicted result={}\n\n'.format(predicted_results[isample]))


    if run_mode == 'query':
        # Ручное тестирование модели и алгоритма восстановления грамматичного текста
        # Грузим конфигурацию модели, веса и т.д.
        with open(config_path, 'r') as f:
            model_config = json.load(f)
            max_inputseq_len = model_config['max_inputseq_len']
            max_outseq_len = model_config['max_outseq_len']
            w2v_path = model_config['w2v_path']
            wordchar2vector_path = model_config['wordchar2vector_path']
            word_dims = model_config['word_dims']
            padding = model_config['padding']
            lemma2id = model_config['lemma2id']
            max_nb_inputs = model_config['max_nb_inputs']

        with open(arch_filepath, 'r') as f:
            model = model_from_json(f.read())

        model.load_weights(weights_path)

        id2lemma = dict((id, lemma) for (lemma, id) in lemma2id.items())
        nb_lemmas = len(lemma2id)

        # Нам нужны таблицы склонения и спряжения
        print('Loading word2lemma dataset...')
        lemma2forms = dict()
        with io.open(os.path.join(data_folder, 'word2lemma.dat'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split('\t')
                if len(tx) == 3:
                    word = tx[0].lower()
                    lemma = tx[1].lower()
                    if lemma in lemma2id:
                        if lemma not in lemma2forms:
                            lemma2forms[lemma] = [word]
                        else:
                            lemma2forms[lemma].append(word)
        lemma2forms[u'ты'] = u'ты тебя тебе тобой'.split()
        lemma2forms[u'я'] = u'я меня мной мне'.split()

        print('Loading the wordchar2vector model {}'.format(wordchar2vector_path))
        wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
        wc2v_dims = len(wc2v.syn0[0])
        print('wc2v_dims={0}'.format(wc2v_dims))

        w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
        w2v_dims = len(w2v.syn0[0])
        print('w2v_dims={0}'.format(w2v_dims))

        word2vec = dict()
        for word in wc2v.vocab:
            v = np.zeros( word_dims )
            v[w2v_dims:] = wc2v[word]
            if word in w2v:
                v[:w2v_dims] = w2v[word]

            word2vec[word] = v

        del w2v
        gc.collect()

        tokenizer = Tokenizer()
        lemmatizer = Lemmatizer()

        while True:
            print('\nEnter two phrases:')
            phrase1 = utils.console_helpers.input_kbd('question:>').lower()
            if len(phrase1) == 0:
                break

            phrase2 = utils.console_helpers.input_kbd('answer:>').lower()
            if len(phrase2) == 0:
                break

            phrases = []
            phrases.append(phrase1)
            phrases.append(phrase2)
            phrase_words = [tokenizer.tokenize(f) for f in phrases]
            phrase_lemmas = [lemmatizer.tokenize(f) for f in phrases]

            input_2grams = set()
            input_1grams = set()
            for phrase in phrases:
                words = tokenizer.tokenize(phrase)
                input_2grams.update(ngrams(words, 2))
                input_1grams.update(words)

            sample = Sample(phrases, phrase_words, phrase_lemmas, u'', [])

            probe_samples = list()
            probe_samples.append(sample)

            for x in generate_rows(max_nb_inputs, nb_lemmas, max_outseq_len, probe_samples, lemma2id, batch_size, 2):
                #print('DEBUG x={}'.format(x['input0'][0, 6]))
                y_pred = model.predict(x=x, verbose=0)
                predicted_lemmas = []
                for lemma_v in y_pred[0]:
                    lemma_index = np.argmax(lemma_v)
                    lemma = id2lemma[lemma_index]
                    if lemma != PAD_WORD:
                        predicted_lemmas.append(lemma)

                print(u'DEBUG predicted_lemmas={}'.format(u' '.join(predicted_lemmas)))
                # Готовим решетку для Витерби, чтобы выбрать оптимальную цепочку словоформ.
                trellis = []
                start = TrellisColumn()
                start.add_cell(InterpreterTrellisNode.create_start())
                trellis.append(start)

                for lemma in predicted_lemmas:
                    column = TrellisColumn()
                    if lemma in lemma2forms:
                        for form in lemma2forms[lemma]:
                            cell = InterpreterTrellisNode(lemma, form)
                            if form not in input_1grams:
                                cell.best_p *= 0.5
                                #print(u'cell[{}].best_p={}'.format(form, cell.best_p))
                            column.add_cell(cell)
                    else:
                        print(u'ERROR: lemma "{}" not in lemma2forms'.format(lemma))
                        exit(1)

                    trellis.append(column)

                end = TrellisColumn()
                end.add_cell(InterpreterTrellisNode.create_end())
                trellis.append(end)

                #print('Start trellis scan...')
                # Идем по столбцам решетки.
                for icolumn in range(1, len(trellis)):
                    column = trellis[icolumn]
                    prev_column = trellis[icolumn-1]
                    for cell in column.cells:
                        best_p = -1.0
                        best_prev = None
                        for prev_cell in prev_column.cells:
                            p12 = cell.best_p*prev_cell.best_p + 1e-6*random.random()
                            n2 = (prev_cell.word, cell.word)
                            if n2 not in input_2grams:
                                p12 *= 0.5

                            if p12 > best_p:
                                best_p = p12
                                best_prev = prev_cell

                        cell.best_prev = best_prev
                        cell.best_p = best_p

                        #print(u'column={} cell={} best_p={} best_prev={}'.format(icolumn, cell.word, best_p, best_prev.word))

                #print('End trellis scan.')

                # Теперь обратный ход - получаем цепочку словоформ.
                cur_cell = end.cells[0]
                selected_cells = []
                while cur_cell is not None:
                    selected_cells.append(cur_cell)
                    cur_cell = cur_cell.best_prev

                selected_cells = selected_cells[::-1][1:-1]
                words = [cell.word for cell in selected_cells]

                print(u'result={}'.format(u' '.join(words)))
                break

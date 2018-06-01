# -*- coding: utf-8 -*-
"""
Тренировка модели для превращения символьной цепочки слова в вектор
действительных чисел фиксированной длины.
 
Реализации RNN и CNN вариантов энкодера, включая комбинации. Реализовано на Keras.
Подробности: https://github.com/Koziev/chatbot/blob/master/PyModels/trainers/README.wordchar2vector.md
"""

from __future__ import print_function

__author__ = "Ilya Koziev"

import six
import numpy as np
import os
import codecs
import random
import json

import keras.callbacks
from keras.layers.core import Activation, RepeatVector, Dense, Masking
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Dropout, Input, Permute, Flatten, Reshape
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import TimeDistributed
from keras.models import Model
from keras.models import model_from_json

# ---------------------------------------------------------------

FILLER_CHAR = u' '  # символ для выравнивания слов по одинаковой длине
BEG_CHAR = u'['  # символ отмечает начало цепочки символов слова
END_CHAR = u']'  # символ отмечает конец цепочки символов слова


def pad_word(word, max_word_len):
    return BEG_CHAR + word + END_CHAR + (max_word_len-len(word))*FILLER_CHAR


def unpad_word(word):
    return word.strip()[1:-1]


def raw_wordset(wordset, max_word_len):
    return [(pad_word(word, max_word_len),pad_word(word, max_word_len)) for word in wordset]


def vectorize_word(word, corrupt_word, X_batch, y_batch, irow, char2index):
    for ich, (ch, corrupt_ch) in enumerate(zip(word, corrupt_word)):
        X_batch[irow, ich] = char2index[corrupt_ch]
        y_batch[irow, ich, char2index[ch]] = True


def generate_rows(wordset, batch_size, char2index, seq_len, mode):
    batch_index = 0
    batch_count = 0

    nb_batches = int(len(wordset)/batch_size)
    nb_chars = len(char2index)

    X_batch = np.zeros((batch_size, seq_len), dtype=np.int32)
    y_batch = np.zeros((batch_size, seq_len, nb_chars), dtype=np.float32)

    shuffled_wordset = list(wordset)
    random.shuffle(shuffled_wordset)

    while True:
        for iword, (word, corrupt_word) in enumerate(shuffled_wordset):
            vectorize_word(word, corrupt_word, X_batch, y_batch, batch_index, char2index )

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1
                # print('mode={} batch_count={}'.format(mode, batch_count))
                if mode == 1:
                    yield ({'input': X_batch}, {'output': y_batch})
                else:
                    yield X_batch

                # очищаем матрицы порции для новой порции
                X_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0


def build_test(wordset, max_word_len, char2index):
    ndata = len(wordset)
    nb_chars = len(char2index)
    X_data = np.zeros((ndata, max_word_len+2), dtype=np.int32)
    y_data = np.zeros((ndata, max_word_len+2, nb_chars), dtype=np.float32)

    for irow, (word, corrupt_word) in enumerate(wordset):
        vectorize_word(word, corrupt_word, X_data, y_data, irow, char2index)

    return X_data, y_data


def build_input(wordset, max_word_len, char2index):
    X,y_unused = build_test(wordset, max_word_len, char2index )
    return X

# -----------------------------------------------------------------

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class VisualizeCallback(keras.callbacks.Callback):
    """
    Класс занимается как визуализацией качества сетки в конце каждой эпохи обучения,
    так и выполняет функции EarlyStopping и ModelCheckpoint колбэков, контролируя
    per install accuracy для валидационного набора.
    """

    def __init__(self, X_test, y_test, model, index2char, weights_path, learning_curve_filename):
        self.epoch = 0
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.index2char = index2char
        self.best_val_acc = -np.inf  # для сохранения самой точной модели
        self.weights_path = weights_path
        self.learning_curve_filename = learning_curve_filename
        self.wait = 0  # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 20
        if os.path.exists(self.learning_curve_filename):
            os.remove(self.learning_curve_filename)

    def decode_char_indeces(self, char_indeces):
        return u''.join([ self.index2char[c] for c in char_indeces ]).strip()

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1

        nb_samples = 0
        nb_errors = 0

        print('')
        for ind in range(len(self.X_test)):
            rowX, rowy = self.X_test[np.array([ind])], self.y_test[np.array([ind])]
            preds = self.model.predict(rowX, verbose=0)

            correct = self.decode_char_indeces(rowy[0,:,:].argmax(axis=-1))
            predicted_char_indeces = preds[0,:,:].argmax(axis=-1)
            guess = self.decode_char_indeces(predicted_char_indeces)

            if ind<10:
                print(colors.ok + '☑ ' + colors.close if correct == guess else colors.fail + '☒ ' + colors.close, end='')
                print(u'wordform={} model_output={}'.format(correct, guess) )

            nb_samples += 1
            if guess!=correct:
                nb_errors += 1

        val_acc = float(nb_samples-nb_errors)/nb_samples

        with open(self.learning_curve_filename, 'a') as wrt:
            wrt.write('{}\t{}\n'.format(self.epoch, val_acc))

        if val_acc > self.best_val_acc:
            print(colors.ok + '\nInstance accuracy improved from {} to {}, saving model to {}\n'.format(self.best_val_acc, val_acc, self.weights_path) + colors.close)
            self.best_val_acc = val_acc
            self.model.save_weights(self.weights_path)
            self.wait = 0
        else:
            print('\nTotal instance accuracy={} did not improve (current best acc={})\n'.format(val_acc, self.best_val_acc))
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_best_accuracy(self):
        return self.best_val_acc

# -----------------------------------------------------------------

class Wordchar2Vector_Trainer(object):
    """
    Класс реализует обучение нейросетевой модели для кодирования слов.
    """
    def __init__(self, arch_type, tunable_char_embeddings, char_dims,
                 model_dir, vec_size, batch_size, seed ):
        self.arch_type = arch_type
        self.tunable_char_embeddings = tunable_char_embeddings
        self.char_dims = char_dims
        self.model_dir = model_dir
        self.vec_size = vec_size
        self.batch_size = batch_size
        self.config_filename = 'wordchar2vector.config'
        self.seed = seed

    def load_words(self, words_filepath):
        # из указанного текстового файла загружаем список слов без повторов
        # и возвращаем его для тренировки или векторизации.
        with codecs.open(words_filepath, 'r', 'utf-8') as rdr:
            return set([line.strip() for line in rdr])

    def train(self, words_filepath, tmp_dir, nb_samples=10000000):
        '''
        Тренируем модель на словах в указанном файле.
        :param words_filepath: путь к plain text utf8 файлу со списком слов (одно слово на строку)
        :param tmp_dir: путь к каталогу, куда будем сохранять всякие сводки по процессу обучения
        для визуализации и прочего контроля
        '''

        # составляем список слов для тренировки и валидации
        known_words = self.load_words(words_filepath)
        print('There are {} known words'.format(len(known_words)))

        max_word_len = max(map(len, known_words))
        seq_len = max_word_len + 2  # 2 символа добавляются к каждому слову для маркировки начала и конца последовательности
        print('max_word_len={}'.format(max_word_len))

        # ограничиваем число слов для обучения и валидации
        if len(known_words)>nb_samples:
            known_words = set(list(known_words)[:nb_samples])

        val_share = 0.3
        random.seed(self.seed)
        train_words = set(filter(lambda z: random.random() > val_share, known_words))
        val_words = set(filter(lambda z: z not in train_words, known_words))

        train_words = raw_wordset(train_words, max_word_len)
        val_words = raw_wordset(val_words, max_word_len)

        print('train set contains {} words'.format(len(train_words)))
        print('val set contains {} words'.format(len(val_words)))

        all_chars = {FILLER_CHAR, BEG_CHAR, END_CHAR}
        for word in known_words:
            all_chars.update(word)

        char2index = {FILLER_CHAR: 0}
        for i, c in enumerate(all_chars):
            if c != FILLER_CHAR:
                char2index[c] = len(char2index)

        index2char = dict([(i, c) for c, i in six.iteritems(char2index)])

        nb_chars = len(all_chars)
        print('nb_chars={}'.format(nb_chars))

        mask_zero = self.arch_type == 'rnn'

        if self.char_dims>0:
            # Символы будут представляться векторами заданной длины,
            # и по мере обучения вектора будут корректироваться для
            # уменьшения общего лосса.
            embedding = Embedding(output_dim=self.char_dims,
                                  input_dim=nb_chars,
                                  input_length=seq_len,
                                  mask_zero=mask_zero,
                                  trainable=True)
        else:
            # 1-hot encoding of characters.
            # длина векторов пользователем не указана, поэтому задаем ее так, что
            # поместилось 1-hot представление.
            self.char_dims = nb_chars

            char_matrix = np.zeros((nb_chars, self.char_dims))
            for i in range(nb_chars):
                char_matrix[i, i] = 1.0

            embedding = Embedding(output_dim=self.char_dims,
                                  input_dim=nb_chars,
                                  input_length=seq_len,
                                  weights=[char_matrix],
                                  mask_zero=mask_zero,
                                  trainable=self.tunable_char_embeddings)

        input_chars = Input(shape=(seq_len,), dtype='int32', name='input')
        encoder = embedding(input_chars)

        print('Building "{}" neural network'.format(self.arch_type))
        if self.arch_type == 'cnn':
            conv_list = []
            merged_size = 0

            nb_filters = 32

            for kernel_size in range(1, 4):
                conv_layer = Conv1D(filters=nb_filters,
                                    kernel_size=kernel_size,
                                    padding='valid',
                                    activation='relu',
                                    strides=1)(encoder)
                conv_layer = GlobalMaxPooling1D()(conv_layer)
                conv_list.append(conv_layer)
                merged_size += nb_filters

            encoder = keras.layers.concatenate(inputs=conv_list)
            encoder = Dense(units=self.vec_size, activation='sigmoid')(encoder)

        elif self.arch_type == 'rnn':
            encoder = recurrent.LSTM(units=self.vec_size, return_sequences=False)(encoder)

        elif self.arch_type == 'bidir_lstm':
            encoder = Bidirectional(recurrent.LSTM(units=int(self.vec_size/2), return_sequences=False))(encoder)

        elif self.arch_type == 'lstm+cnn':
            conv_list = []
            merged_size = 0

            rnn_size = self.vec_size
            conv_list.append(recurrent.LSTM(units=rnn_size, return_sequences=False)(encoder))
            merged_size += rnn_size

            nb_filters = 32
            for kernel_size in range(1, 4):
                conv_layer = Conv1D(filters=nb_filters,
                                    kernel_size=kernel_size,
                                    padding='valid',
                                    activation='relu',
                                    strides=1)(encoder)
                conv_layer = GlobalMaxPooling1D()(conv_layer)
                conv_list.append(conv_layer)
                merged_size += nb_filters

            encoder = keras.layers.concatenate(inputs=conv_list)
            encoder = Dense(units=self.vec_size, activation='sigmoid')(encoder)

        elif self.arch_type == 'lstm(cnn)':

            conv_list = []
            merged_size = 0

            nb_filters = 32
            rnn_size = nb_filters

            for kernel_size in range(1, 4):
                conv_layer = Conv1D(filters=nb_filters,
                                    kernel_size=kernel_size,
                                    padding='valid',
                                    activation='relu',
                                    strides=1,
                                    name='shared_conv_{}'.format(kernel_size))(encoder)

                conv_layer = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer)
                conv_layer = recurrent.LSTM(rnn_size, return_sequences=False)(conv_layer)

                conv_list.append(conv_layer)
                merged_size += rnn_size

            encoder = keras.layers.concatenate(inputs=conv_list)
            encoder = Dense(units=self.vec_size, activation='sigmoid')(encoder)

        else:
            raise RuntimeError('Unknown architecture of neural net: {}'.format(self.arch_type))

        decoder = RepeatVector(seq_len)(encoder)
        decoder = recurrent.LSTM(self.vec_size, return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(nb_chars, activation='softmax'), name='output')(decoder)

        model = Model(inputs=input_chars, outputs=decoder)
        model.compile(loss='categorical_crossentropy', optimizer='nadam')

        keras.utils.plot_model(model,
                               to_file=os.path.join(self.model_dir, 'wordchar2vector.arch.png'),
                               show_shapes=False,
                               show_layer_names=True)

        weigths_path = os.path.join(self.model_dir, 'wordchar2vector.model')
        arch_filepath = os.path.join(self.model_dir, 'wordchar2vector.arch')

        model_config = {
            'max_word_len': max_word_len,
            'seq_len': seq_len,
            'char2index': char2index,
            'FILLER_CHAR': FILLER_CHAR,
            'BEG_CHAR': BEG_CHAR,
            'END_CHAR': END_CHAR,
            'arch_filepath': arch_filepath,
            'weights_path': weigths_path,
            'vec_size': self.vec_size
        }

        with open(os.path.join(self.model_dir, self.config_filename), 'w') as f:
            json.dump(model_config, f)


        # model_checkpoint = ModelCheckpoint( weigths_path,
        #                                    monitor='val_loss',
        #                                    verbose=1,
        #                                    save_best_only=True,
        #                                    mode='auto')
        #
        # early_stopping = EarlyStopping(monitor='val_loss',
        #                                patience=20,
        #                                verbose=1,
        #                                mode='auto')


        X_viz, y_viz = build_test(list(val_words)[0:1000], max_word_len, char2index)

        learning_curve_filename = os.path.join(tmp_dir,
                                               'learning_curve__{}_vecsize={}_tunable_char_embeddings={}_chardims={}_batchsize={}_seed={}.csv'.format(
                                                   self.arch_type, self.vec_size, self.tunable_char_embeddings,
                                                   self.char_dims, self.batch_size, self.seed))
        visualizer = VisualizeCallback(X_viz, y_viz, model, index2char, weigths_path,
                                       learning_curve_filename)

        # csv_logger = CSVLogger(learning_curve_filename, append=True, separator='\t')

        hist = model.fit_generator(generator=generate_rows(train_words, self.batch_size, char2index, seq_len, 1),
                                   steps_per_epoch=int(len(train_words) / self.batch_size),
                                   epochs=1000,
                                   verbose=1,
                                   callbacks=[visualizer],  # csv_logger, model_checkpoint, early_stopping],
                                   validation_data=generate_rows(val_words, self.batch_size, char2index, seq_len, 1),
                                   validation_steps=int(len(val_words) / self.batch_size),
                                   )
        print('Training complete, best_accuracy={}'.format(visualizer.get_best_accuracy()))

        # Загружаем наилучшее состояние модели
        model.load_weights(weigths_path)

        # Создадим модель с урезанным до кодирующей части графом.
        model = Model(inputs=input_chars, outputs=encoder)

        # Сохраним эту модель
        with open(arch_filepath, 'w') as f:
            f.write(model.to_json())

        # Пересохраним веса усеченной модели
        model.save_weights(weigths_path)

    def vectorize(self, words_filepath, result_path):
        '''
        Векторизуем слова из списка (указан путь к текстовому файлу со списком).
        Созданные векторы слов сохраняем по указанному пути как текстовый файл
        в w2v-совместимом формате.
        '''

        # читаем конфиг модели
        with open(os.path.join(self.model_dir, self.config_filename), 'r') as f:
            model_config = json.load(f)

        # грузим готовую модель
        with open(model_config['arch_filepath'], 'r') as f:
            model = model_from_json(f.read())

        model.load_weights(model_config['weights_path'])

        self.vec_size = model_config['vec_size']
        self.max_word_len = model_config['max_word_len']
        self.char2index = model_config['char2index']

        output_words = self.load_words(words_filepath)
        nb_words = len(output_words)
        print(u'{} words will be vectorized and stored to {}'.format(nb_words, result_path))

        with codecs.open( result_path, 'w', 'utf-8') as wrt:
            wrt.write('{} {}\n'.format(nb_words, self.vec_size))

            nb_batch = int(nb_words/self.batch_size) + (0 if (nb_words%self.batch_size) == 0 else 1)
            wx = list(output_words)
            words = raw_wordset(wx, self.max_word_len)

            words_remainder = nb_words
            word_index=0
            while words_remainder>0:
                print('words_remainder={:<10d}'.format(words_remainder), end='\r')
                nw = min( self.batch_size, words_remainder )
                batch_words = words[word_index:word_index+nw]
                X_data = build_input(batch_words, self.max_word_len, self.char2index)
                y_pred = model.predict( x=X_data, batch_size=nw, verbose=0 )

                for iword, (word,corrupt_word) in enumerate(batch_words):
                    word_vect = y_pred[iword, :]
                    naked_word = unpad_word(word)
                    wrt.write(u'{} {}\n'.format(naked_word, u' '.join([str(x) for x in word_vect])))

                word_index += nw
                words_remainder -= nw


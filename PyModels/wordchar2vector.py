# -*- coding: utf-8 -*-
'''
Тренировка модели для превращения символьной цепочки слова в вектор.
RNN и CNN варианты энкодера.

Список слов, для которых строится модель, читается из файла ../tmp/words.txt
и должен быть предварительно сформирован скриптом prepare_wordchar_dataset.py

(c) by Koziev Ilya inkoziev@gmail.com
'''

from __future__ import print_function
import argparse
from keras.layers.core import Activation, RepeatVector, Dense, Masking
#from keras.layers.wrappers import TimeDistributed
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Dropout, Input, Permute, Flatten, Reshape
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import TimeDistributed
from keras.models import Model
from keras.callbacks import CSVLogger
import numpy as np
import os
import codecs
import random


ARCH_TYPE = 'rnn' # архитектура модели: cnn - сверточный энкодер, rnn - рекуррентный энкодер
tunable_char_embeddings = False # делать ли настраиваемые векторы символов (True) или 1-hot (False)
batch_size = 250

parser = argparse.ArgumentParser(description='Training the wordchar2vector embeddings for words')
parser.add_argument('--input_file', default='../tmp/known_words.txt', help='input text file with words to be processed')
parser.add_argument('--out_file', default='../tmp/wordchar2vector.dat', help='output text file containing with word vectors in word2vec text format')
parser.add_argument('--model_dir', help='folder with model files', default='../tmp')
parser.add_argument('--train', default=0, type=int )
parser.add_argument('--vectorize', default=0, type=int )
parser.add_argument('--dims', default=56, type=int )

args = parser.parse_args()

model_dir = args.model_dir # каталог для файлов модели - при тренировке туда записываются, при векторизации - оттуда загружаются
input_path = args.input_file # из этого файла прочитаем список слов, на которых учится модель
out_file = args.out_file # в этот файл будет сохранены векторы слов в word2vec-совместимом формате
do_train = args.train # тренировать ли модель с нуля
do_vectorize = args.vectorize # векторизовать ли входной список слов
vec_size = args.dims # размер вектора представления слова для тренировки модели

# -------------------------------------------------------------------

if not do_train and not do_vectorize:
    while True:
        a1 = raw_input('0-train model\n1-calculate embeddings using pretrained model\n[0/1]: ')
        if a1=='0':
            do_train = True
            do_vectorize = True
            print('Training the model...')
            break
        elif a1=='1':
            do_train = False
            do_vectorize = True
            print('Calculating the word embeddings...')
            break
        else:
            print('Unrecognized choice "{}"'.format(a1))


FILLER_CHAR = u' '
BEG_CHAR = u'['
END_CHAR = u']'

# ---------------------------------------------------------------


def pad_word( word, max_word_len ):
    return BEG_CHAR + word + END_CHAR + (max_word_len-len(word))*FILLER_CHAR


def unpad_word(word):
    return word.strip()[1:-1]


def raw_wordset( wordset, max_word_len ):
    return [ (pad_word(word, max_word_len),pad_word(word, max_word_len)) for word in wordset ]


known_words = set()
with codecs.open(input_path, 'r', 'utf-8') as rdr:
    line_count = 0
    for line0 in rdr:
        word = line0.strip()
        known_words.add(word)

print('There are {} known words'.format(len(known_words)))

max_word_len = max( map(len,known_words) )
seq_len = max_word_len+2 # 2 символа добавляются к каждому слову для маркировки начала и конца последовательности
print('max_word_len={}'.format(max_word_len))

val_share = 0.3
train_words = set( filter( lambda z:random.random()>val_share, known_words ) )
val_words = set( filter( lambda z:z not in train_words, known_words) )

train_words = raw_wordset(train_words, max_word_len)
val_words = raw_wordset(val_words, max_word_len )

print('train set contains {} words'.format(len(train_words)))
print('val set contains {} words'.format(len(val_words)))

all_chars = { FILLER_CHAR, BEG_CHAR, END_CHAR }
for word in known_words:
    all_chars.update(word)

char2index = { FILLER_CHAR:0 }
for i,c in enumerate(all_chars):
    if c!=FILLER_CHAR:
        char2index[c] = len(char2index)

index2char = dict( [ (i,c) for c,i in char2index.iteritems() ])

nb_chars = len(all_chars)
print('nb_chars={}'.format(nb_chars))

# -----------------------------------------------------------------

def vectorize_word( word, corrupt_word, X_batch, y_batch, irow, char2index ):
    for ich, (ch, corrupt_ch) in enumerate(zip(word, corrupt_word)):
        X_batch[irow, ich] = char2index[corrupt_ch]
        y_batch[irow, ich, char2index[ch]] = True


def generate_rows( wordset, batch_size, char2index, mode ):
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


def build_test( wordset, max_word_len, char2index ):
    ndata = len(wordset)
    nb_chars = len(char2index)
    X_data = np.zeros((ndata, max_word_len+2), dtype=np.int32)
    y_data = np.zeros((ndata, max_word_len+2, nb_chars), dtype=np.float32)

    for irow, (word, corrupt_word) in enumerate(wordset):
        vectorize_word(word, corrupt_word, X_data, y_data, irow, char2index)

    return (X_data, y_data)


def build_input( wordset, max_word_len, char2index ):
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

    def __init__(self, X_test, y_test, model, index2char, weights_path):
        self.epoch = 0
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.index2char = index2char
        self.best_val_acc = 0.0 # для сохранения самой точной модели
        self.weights_path = weights_path
        self.wait = 0 # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 20


    def decode_char_indeces(self, char_indeces):
        return u''.join([ self.index2char[c] for c in char_indeces ]).strip()


    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1

        nb_samples = 0
        nb_errors = 0

        print('')
        #with open( self.output_path, 'a' ) as fsamples:
            #fsamples.write( u'\n' + '='*50 + u'\nepoch=' + str(self.epoch) + u'\n' );
        for ind in range(len(self.X_test)):
            rowX, rowy = self.X_test[np.array([ind])], self.y_test[np.array([ind])]
            preds = self.model.predict(rowX, verbose=0)

            correct = self.decode_char_indeces(rowy[0,:,:].argmax(axis=-1))
            #guess = self.ctable.decode(preds[0], calc_argmax=False)

            predicted_char_indeces = preds[0,:,:].argmax(axis=-1)
            guess = self.decode_char_indeces(predicted_char_indeces)

            if ind<10:
                print(colors.ok + '☑ ' + colors.close if correct == guess else colors.fail + '☒ ' + colors.close, end='')
                print(u'wordform={} model_output={}'.format(correct, guess) )

            #fsamples.write( (correct + u' ==> ' + guess + u'\n').encode('utf-8') )
            nb_samples += 1
            if guess!=correct:
                nb_errors += 1

        val_acc = float(nb_samples-nb_errors)/nb_samples
        if val_acc>self.best_val_acc:
            print(colors.ok +'\nInstance accuracy improved from {} to {}, saving model to {}\n'.format(self.best_val_acc, val_acc, self.weights_path)+ colors.close)
            self.best_val_acc = val_acc
            self.model.save_weights(self.weights_path)
            self.wait = 0
        else:
            print('\nTotal instance accuracy={} did not improve\n'.format(val_acc))
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

# -----------------------------------------------------------------

mask_zero = ARCH_TYPE=='rnn'

word_dims = nb_chars

char_matrix = np.zeros((nb_chars, word_dims))
for i in range(nb_chars):
    char_matrix[i, i] = 1.0

embedding = Embedding(output_dim=word_dims,
                      input_dim=nb_chars,
                      input_length=seq_len,
                      weights=[char_matrix],
                      mask_zero=mask_zero,
                      trainable=tunable_char_embeddings)



input_chars = Input(shape=(seq_len,), dtype='int32', name='input')
encoder = embedding(input_chars)


if ARCH_TYPE=='cnn':
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
    encoder = Dense(units=vec_size, activation='sigmoid')(encoder)
elif ARCH_TYPE=='rnn':
    encoder = recurrent.LSTM(units=vec_size, return_sequences=False)(encoder)
    #encoder = Bidirectional(recurrent.LSTM(units=int(vec_size/2), return_sequences=False))(encoder)
    #encoder = Dense(units=vec_size, activation='relu')(encoder)
    #encoder = Dense(units=vec_size, activation='relu')(encoder)
else:
    raise RuntimeError()


decoder = RepeatVector(seq_len)(encoder)
decoder = recurrent.LSTM(vec_size, return_sequences=True)(decoder)
decoder = TimeDistributed(Dense(nb_chars, activation='softmax'), name='output')(decoder)

model = Model(inputs=input_chars, outputs=decoder)
model.compile(loss='categorical_crossentropy', optimizer='nadam')
#model.compile(loss='categorical_crossentropy', optimizer='adamax')

keras.utils.plot_model(model, to_file='../tmp/wordchar2vector.arch.png', show_shapes=False, show_layer_names=True)

weigths_path = os.path.join(model_dir, 'wordcharsvector.model')


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


X_viz, y_viz = build_test( list(val_words)[0:1000], max_word_len, char2index )

visualizer = VisualizeCallback(X_viz, y_viz, model, index2char, weigths_path)

learning_curve_filename = os.path.join( model_dir, 'learning_curve_{}_vecsize={}_tunable_char_embeddings={}.csv'.format(ARCH_TYPE, vec_size, tunable_char_embeddings) )
csv_logger = CSVLogger(learning_curve_filename, append=True, separator='\t')


if do_train:
    hist = model.fit_generator(generator=generate_rows(train_words, batch_size, char2index, 1),
                               steps_per_epoch=int(len(train_words)/batch_size),
                               epochs=200,
                               verbose=1,
                               callbacks=[visualizer, csv_logger], #model_checkpoint, early_stopping],
                               validation_data=generate_rows( val_words, batch_size, char2index, 1),
                               validation_steps=int(len(val_words)/batch_size),
                               )

    model.load_weights(weigths_path)

# --------------------------------------------------------------------

# генерируем векторы для слов и сохраняем их в файле для
# последующего использования.

# Создадим модель с урезанным до кодирующей части графом.
model = Model(inputs=input_chars, outputs=encoder)

# Сохраним эту модель
with open(os.path.join(model_dir, 'wordchar2vector.arch'), 'w') as f:
    f.write(model.to_json())

if do_train:
    model.save_weights(weigths_path)
else:
    model.load_weights(weigths_path)

if do_vectorize:
    output_words = set(known_words)
    nb_words = len(output_words)

    with codecs.open( out_file, 'w', 'utf-8') as wrt:
        wrt.write('{} {}\n'.format(nb_words, vec_size))

        nb_batch = int(nb_words/batch_size) + (0 if (nb_words%batch_size)==0 else 1)
        wx = list(output_words)
        words = raw_wordset( wx, max_word_len )

        words_remainder = nb_words
        word_index=0
        while words_remainder>0:
            print('words_remainder={}        '.format(words_remainder), end='\r')
            nw = min( batch_size, words_remainder )
            batch_words = words[word_index:word_index+nw]
            X_data = build_input(batch_words, max_word_len, char2index)
            y_pred = model.predict( x=X_data, batch_size=nw, verbose=0 )

            for iword,(word,corrupt_word) in enumerate(batch_words):
                word_vect = y_pred[iword,:]
                naked_word = unpad_word(word)
                wrt.write(u'{} {}\n'.format(naked_word, u' '.join([str(x) for x in word_vect]) ))

            word_index += nw
            words_remainder -= nw

print('\nDone.')

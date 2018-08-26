# -*- coding: utf-8 -*-
'''
Посимвольная генерация первого слова ответа.
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import gc
import itertools
import json
import os

import gensim
import keras.callbacks
import numpy as np
import pandas as pd
import tqdm
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import recurrent
from keras.layers.core import RepeatVector, Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import add, multiply
from keras.layers import Lambda

from keras.models import Model
from sklearn.model_selection import train_test_split

from utils.tokenizer import Tokenizer

input_path = '../data/premise_question_answer.csv'
tmp_folder = '../tmp'
data_folder = '../data'

#NET_ARCH = 'lstm'
#NET_ARCH = 'lstm+cnn'
NET_ARCH = 'lstm(cnn)'

encoder_arch = 'merge'
#encoder_arch = 'muladd'

#BATCH_SIZE = 1000
BATCH_SIZE = 400

# -------------------------------------------------------------------

PAD_WORD = u''
PAD_CHAR = u'\r'

# -------------------------------------------------------------------------

# Слева добавляем пустые слова
def pad_wordseq(words, n):
    return list( itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


# Справа добавляем пустые слова
def rpad_wordseq(words, n):
    return list( itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def rpad_charseq(s, n):
    return s+PAD_CHAR*max(0, n-len(s))


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def generate_rows(sequences, targets, batch_size, mode, char2id):
    batch_index = 0
    batch_count = 0

    X1_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, max_outword_len, output_dims), dtype=np.bool)

    while True:
        for irow, (seq, target) in enumerate(itertools.izip(sequences, targets)):
            vectorize_words(seq[0], X1_batch, batch_index, word2vec)
            vectorize_words(seq[1], X2_batch, batch_index, word2vec)
            for ichar, c in enumerate(target):
                y_batch[batch_index, ichar, char2id[c]] = True

            batch_index += 1

            if batch_index == batch_size:
                batch_count += 1

                if mode == 1:
                    yield ({'input_words1': X1_batch, 'input_words2': X2_batch}, {'output': y_batch})
                else:
                    yield {'input_words1': X1_batch, 'input_words2': X2_batch}

                # очищаем матрицы порции для новой порции
                X1_batch.fill(0)
                X2_batch.fill(0)
                y_batch.fill(0)
                batch_index = 0

# -------------------------------------------------------------------


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


class VisualizeCallback(keras.callbacks.Callback):

    def __init__(self, val_input, val_output, model, wc2v, char2id, weights_path):
        self.epoch = 0
        self.weights_path = weights_path
        self.val_input = val_input
        self.val_output = val_output
        self.model = model
        self.wc2v = wc2v
        self.char2id = char2id
        self.id2char = dict([(i,c) for c,i in char2id.iteritems()])
        self.best_val_acc = 0.0 # для сохранения самой точной модели
        self.nb_epochs_wo_improvements = 0
        self.nb_early_stopping = 10
        self.stopped_epoch = 0

    def decode_str(self, y):
        s = []
        for char_v in y:
            char_index = np.argmax(char_v)
            c = self.id2char[char_index]
            s.append(c)

        return u''.join(s).strip()

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        nval = len(self.val_input)

        print('\n')

        # восстанавливаем сгенерированные символьные цепочки ответов, сравниваем с
        # требуемыми цепочками.
        nb_errors = 0
        nb_samples = 0

        # Счетчик напечатанных строк, сгенерированных моделью
        nb_shown = 0

        nb_steps = int(nval/BATCH_SIZE)

        for step,batch in enumerate(generate_rows(self.val_input, self.val_output, BATCH_SIZE, 1, self.char2id)):
            if step == nb_steps:
                break

            y_batch = batch[1]['output']
            y_pred = model.predict_on_batch(batch[0])

            for iy in range(len(y_pred)):
                target_chars = self.decode_str(y_batch[iy])
                predicted_chars = self.decode_str(y_pred[iy])

                nb_samples += 1
                if predicted_chars != target_chars:
                    nb_errors += 1

                if nb_shown < 10:
                    print(
                        colors.ok+'☑ '+colors.close if predicted_chars == target_chars else colors.fail + '☒ ' + colors.close,
                        end='')

                    print(u'true={:30s} model={}'.format(target_chars, predicted_chars))
                    nb_shown += 1

        val_acc = (nb_samples-nb_errors)/float(nb_samples)
        if val_acc>self.best_val_acc:
            print(colors.ok+'\nInstance accuracy improved from {} to {}, saving model to {}\n'.format(self.best_val_acc, val_acc, self.weights_path)+ colors.close)
            self.best_val_acc = val_acc
            self.model.save_weights(self.weights_path)
            self.nb_epochs_wo_improvements = 0
        else:
            self.nb_epochs_wo_improvements += 1
            print('\nInstance accuracy={} did not improve ({} epochs since last improvement)\n'.format(val_acc, self.nb_epochs_wo_improvements))
            if self.nb_epochs_wo_improvements >= self.nb_early_stopping:
                print('Early stopping.')
                self.model.stop_training = True
                self.stopped_epoch = self.epoch

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Best instance accuracy={}'.format(self.best_val_acc))


# -------------------------------------------------------------------

RUN_MODE = 'train'


wordchar2vector_path = os.path.join(data_folder,'wordchar2vector.dat')
print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])
print('wc2v_dims={0}'.format(wc2v_dims))

# --------------------------------------------------------------------------

df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
print('samples.count={}'.format(df.shape[0]))

max_inputseq_len = 0
max_outword_len = 0 # максимальная длина выходного слова в символах
all_words = set([PAD_WORD])
all_chars = set([PAD_CHAR])

tokenizer = Tokenizer()
for i,record in df.iterrows():
    answer = record['answer']
    if answer not in [u'да', u'нет']:
        all_chars.update( answer )

        words = tokenizer.tokenize(answer)
        max_outword_len = max(max_outword_len, len(words[0]))

        for phrase in [record['premise'], record['question']]:
            all_chars.update( phrase )
            words = tokenizer.tokenize(phrase)
            all_words.update(words)
            max_inputseq_len = max( max_inputseq_len, len(words) )


for word in wc2v.vocab:
    all_words.add(word)

print('max_inputseq_len={}'.format(max_inputseq_len))
print('max_outword_len={}'.format(max_outword_len))

char2id = dict( [(c,i) for i,c in enumerate( itertools.chain([PAD_CHAR], filter(lambda z:z!=PAD_CHAR,all_chars)))] )

nb_chars = len(all_chars)
print('nb_chars={}'.format(nb_chars))

# --------------------------------------------------------------------------

w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=8.bin'
#w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=48.txt'
#w2v_path = '/home/eek/polygon/WordSDR2/sdr.dat'
#w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=128.txt'
#w2v_path = r'f:\Word2Vec\word_vectors_cbow=1_win=5_dim=32.txt'
#w2v_path = os.path.join(tmp_folder,'w2v.CBOW=1_WIN=5_DIM=32.model')
print( 'Loading the w2v model {}'.format(w2v_path) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('.txt'))
w2v_dims = len(w2v.syn0[0])
print('w2v_dims={0}'.format(w2v_dims))

word_dims = w2v_dims+wc2v_dims


word2vec = dict()
for word in wc2v.vocab:
    v = np.zeros( word_dims )
    v[w2v_dims:] = wc2v[word]
    if word in w2v:
        v[:w2v_dims] = w2v[word]

    word2vec[word] = v

#del w2v
#del wc2v
gc.collect()
# -------------------------------------------------------------------

nb_filters = 128
rnn_size = word_dims

final_merge_size = 0

# --------------------------------------------------------------------------------

# сохраним конфиг модели, чтобы ее использовать в чат-боте
model_config = {
                'max_inputseq_len': max_inputseq_len,
                'max_outword_len': max_outword_len,
                'w2v_path': w2v_path,
                'wordchar2vector_path': wordchar2vector_path,
                'PAD_WORD': PAD_WORD,
                'PAD_CHAR': ord(PAD_CHAR),
                'model_folder': tmp_folder,
                'word_dims': word_dims
               }

with open(os.path.join(tmp_folder,'qa_chargenerator.config'), 'w') as f:
    json.dump(model_config, f)

# ------------------------------------------------------------------

# В этих файлах будем сохранять натренированную сетку
arch_filepath = os.path.join(tmp_folder, 'qa_chargenerator.arch')
weights_path = os.path.join(tmp_folder, 'qa_chargenerator.weights')

# ------------------------------------------------------------------

if RUN_MODE == 'train':

    print('Building the NN computational graph...')

    words_net1 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words1')
    words_net2 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words2')

    conv1 = []
    conv2 = []

    repr_size = 0

    if NET_ARCH == 'lstm':
        # Энкодер на базе LSTM, на выходе которого получаем вектор с упаковкой слов
        # каждого из входных предложений. Сетка сделана общей для предпосылки и вопроса,
        # так как такое усреднение улучшает качество в сравнении с вариантом раздельных
        # сеток для каждого из предложений.
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size, return_sequences=False))
        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)
        repr_size += rnn_size*2

    if NET_ARCH == 'lstm+cnn':
        shared_words_rnn = Bidirectional(recurrent.LSTM(rnn_size, return_sequences=False))
        encoder_rnn1 = shared_words_rnn(words_net1)
        encoder_rnn2 = shared_words_rnn(words_net2)

        conv1.append(encoder_rnn1)
        conv2.append(encoder_rnn2)

        repr_size += rnn_size*2

        # добавляем входы со сверточными слоями
        for kernel_size in range(2, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            pooler = GlobalAveragePooling1D()

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += nb_filters

    if NET_ARCH == 'lstm(cnn)':
        # Гибридная архитектура - рекуррентные слои поверх сверточных
        #for kernel_size, nb_filters in [(1, 16), (2, 50), (3, 100), (4, 200)]:
        for kernel_size in range(2, 4):
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1)

            rnn_size = nb_filters*2
            rnn = Bidirectional(recurrent.LSTM(rnn_size, return_sequences=False))

            conv_layer1 = conv(words_net1)
            conv_layer1 = rnn(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = rnn(conv_layer2)
            conv2.append(conv_layer2)

            repr_size += rnn_size*2

    encoder_size = repr_size

    if encoder_arch == 'merge':
        encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2)))
        encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_merged)
    elif encoder_arch == 'muladd':
        encoder1 = keras.layers.concatenate(inputs=conv1)
        encoder2 = keras.layers.concatenate(inputs=conv2)

        addition = add([encoder1, encoder2])
        minus_y1 = Lambda(lambda x: -x, output_shape=(encoder_size,))(encoder1)
        mul = add([encoder2, minus_y1])
        mul = multiply([mul, mul])

        #encoder_final = keras.layers.concatenate(inputs=[encoder1, mul, addition, encoder2])
        encoder_final = keras.layers.concatenate(inputs=[mul, addition])
        encoder_final = Dense(units=int(encoder_size), activation='relu')(encoder_final)

    # Теперь добавляем вторую часть сетки - посимвольный декодер слова.
    output_dims = nb_chars
    decoder = encoder_final

    #decoder = Dense(units=encoder_size, activation='sigmoid')(encoder_final)
#    decoder = Dense(units=encoder_size, activation='relu')(decoder)
#    decoder = Dense(units=encoder_size, activation='relu')(decoder)
    decoder = RepeatVector(max_outword_len)(decoder)
    decoder = recurrent.LSTM(encoder_size, return_sequences=True)(decoder)
    decoder = TimeDistributed(Dense(nb_chars, activation='softmax'), name='output')(decoder)

    model = Model(inputs=[words_net1, words_net2], outputs=decoder)
    model.compile(loss='categorical_crossentropy', optimizer='nadam')

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())


    input_data = []
    output_data = []

    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extract phrases'):
        premise = row['premise']
        question = row['question']
        answer = row['answer']

        if answer not in [u'да', u'нет']:
            premise_words = rpad_wordseq( tokenizer.tokenize(premise), max_inputseq_len )
            question_words = rpad_wordseq( tokenizer.tokenize(question), max_inputseq_len )

            answer_words = tokenizer.tokenize(answer)

            input_data.append( (premise_words, question_words, premise, question) )
            output_data.append( rpad_charseq(answer_words[0], max_outword_len) )

    SEED = 123456
    TEST_SHARE = 0.2
    train_input, val_input, train_output, val_output = train_test_split( input_data, output_data, test_size=TEST_SHARE, random_state=SEED )

    batch_size = BATCH_SIZE

    nb_train_patterns = len(train_input)
    nb_valid_patterns = len(val_input)

    print('Start training using {} patterns for training, {} for validation...'.format(nb_train_patterns, nb_valid_patterns))

    monitor_metric = 'val_loss'

    #model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
    #                                   verbose=1, save_best_only=True, mode='auto')
    #early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

    viz = VisualizeCallback(val_input, val_output, model, wc2v, char2id, weights_path)

    callbacks = [viz]


    hist = model.fit_generator(generator=generate_rows(train_input, train_output, batch_size, 1, char2id),
                               steps_per_epoch=int(nb_train_patterns/batch_size),
                               epochs=200,
                               verbose=1,
                               callbacks=callbacks,
                               validation_data=generate_rows( val_input, val_output, batch_size, 1, char2id),
                               validation_steps=int(nb_valid_patterns/batch_size)
                               )

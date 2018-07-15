# -*- coding: utf-8 -*-
'''
Тренировка сеточных моделей для генерации wc2v вектора первого, второго etc. слов
ответа для чатбота https://github.com/Koziev/chatbot.
'''

from __future__ import print_function
from __future__ import division  # for python2 compatibility

import pandas as pd
import numpy as np
import sys
import gc
import gensim
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from keras.layers.wrappers import TimeDistributed
from keras.models import model_from_json
from keras.layers.merge import concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.layers.core import Activation, RepeatVector, Dense, Masking
from keras.layers.wrappers import Bidirectional
from keras.layers import Input
from keras.layers import BatchNormalization
import keras.callbacks
from keras.layers import recurrent

import itertools
import tqdm
import os
import json

from utils.tokenizer import Tokenizer

input_path = '../data/premise_question_answer.csv'
tmp_folder = '../tmp'
data_folder = '../data'

# Вариант архитектуры нейросетки:
# lstm - один общий bidir LSTM слой упаковывает слова предпосылки и вопроса в векторы. Затем два
#        получившихся вектора поступают на афинные слои классификатора.
# lstm+cnn - параллельно с LSTM слоями из первого варианта работают сверточные слои. Затем
#        все векторы соединяются и поступают на классификатор.
# lstm(cnn) - сначала группа сверточных слоев выделяет словосочетания, затем после пулинга
#        эти признаки поступают на LSTM и далее на классификатор.
#NET_ARCH = 'lstm'
#NET_ARCH = 'lstm+cnn'
NET_ARCH = 'lstm(cnn)'

#BATCH_SIZE = 1000
batch_size = 50

# -------------------------------------------------------------------

PAD_WORD = u''


# Слева добавляем пустые слова
def pad_wordseq(words, n):
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))


# Справа добавляем пустые слова
def rpad_wordseq(words, n):
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))


def vectorize_words(words, M, irow, word2vec):
    for iword,word in enumerate(words):
        if word in word2vec:
            M[irow, iword, :] = word2vec[word]


def generate_rows(sequences, targets, batch_size, wc2v, mode):
    batch_index = 0
    batch_count = 0

    wc2v_dim = len(wc2v.syn0[0])

    X1_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    X2_batch = np.zeros((batch_size, max_inputseq_len, word_dims), dtype=np.float32)
    y_batch = np.zeros((batch_size, wc2v_dim), dtype=np.float32)

    while True:
        for irow, (seq, target_word) in enumerate(itertools.izip(sequences, targets)):
            vectorize_words(seq[0], X1_batch, batch_index, word2vec)
            vectorize_words(seq[1], X2_batch, batch_index, word2vec)
            y_batch[batch_index, :] = wc2v[target_word]

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

    def __init__(self, val_input, val_output, model, wc2v, weights_path, itarget, batch_size):
        self.epoch = 0
        self.itarget = itarget
        self.weights_path = weights_path
        self.val_input = val_input
        self.val_output = val_output
        self.model = model
        self.wc2v = wc2v
        self.best_val_acc = 0.0 # для сохранения самой точной модели
        self.wait = 0 # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 10
        self.batch_size = batch_size

        # для быстрого поиска слова по вектору попробуем использовать kd-tree,
        # благо размерность векторов модели wc2v относительно небольшая.
        vectors = []
        self.wc_words = []

        for w, v in wc2v.vocab.iteritems():
            self.wc_words.append(w)
            vectors.append(wc2v[w])

        self.kdtree = KDTree(vectors)


    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        nval = min(1000, len(self.val_input))

        print('\n')

        nb_errors = 0
        nb_samples = 0

        nb_shown = 0

        nb_steps = nval//self.batch_size

        for step, batch in enumerate(generate_rows(self.val_input, self.val_output, self.batch_size, self.wc2v, 1)):
            if step == nb_steps:
                break

            y_batch = batch[1]['output']
            y_pred = model.predict_on_batch(batch[0])
            ny = y_pred[0].shape[0]

            for iy in range(min(len(y_batch), len(y_pred))):
                expected_wordchar_vector = y_batch[iy]
                predicted_wordchar_vector = y_pred[iy]

                # ищем слово, максимально близкое для этого вектора
                _, iwords = self.kdtree.query([predicted_wordchar_vector])
                predicted_closest_word = self.wc_words[iwords[0]]

                _, iwords = self.kdtree.query([expected_wordchar_vector])
                expected_closest_word = self.wc_words[iwords[0]]

                predict_ok = True
                if predicted_closest_word != expected_closest_word:
                    nb_errors += 1
                    predict_ok = False
                nb_samples += 1

                if nb_shown < 10:
                    print(
                        colors.ok + '☑ ' + colors.close if predict_ok else colors.fail + '☒ ' + colors.close,
                        end='')

                    print(u'true={} model={}'.format(expected_closest_word.ljust(30), predicted_closest_word.ljust(30)))
                    nb_shown += 1

        print('\n')
        val_acc = (nb_samples-nb_errors)/float(nb_samples)
        if val_acc > self.best_val_acc:
            print(colors.ok +'\nInstance accuracy for itarget={} improved from {} to {}, saving model to {}\n'.format(self.itarget, self.best_val_acc, val_acc, self.weights_path)+ colors.close)
            self.best_val_acc = val_acc
            self.model.save_weights(self.weights_path)
            self.wait = 0
        else:
            print('\nTotal instance for itarget={} accuracy={} did not improve\n'.format(self.itarget, val_acc))
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


# -------------------------------------------------------------------

RUN_MODE = ''

while True:
    print('t - train ')
    print('q - query')
    a1 = raw_input(':> ')

    if a1 == 't':
        RUN_MODE = 'train'
        print('Train')
        break
    elif a1 == 'q':
        RUN_MODE = 'query'
        print('Query')
        break
    else:
        print('Unrecognized choice "{}"'.format(a1))


# --------------------------------------------------------------------------

wordchar2vector_path = os.path.join(data_folder,'wordchar2vector.dat')
print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])
print('wc2v_dims={0}'.format(wc2v_dims))

# --------------------------------------------------------------------------

df = pd.read_csv(input_path, encoding='utf-8', delimiter='\t', quoting=3)
print('samples.count={}'.format(df.shape[0]))

max_inputseq_len = 0
max_nb_words = 0 # макс. число слов в ответе
all_words = set([PAD_WORD])

tokenizer = Tokenizer()
for i,record in df.iterrows():
    answer = record['answer']
    words = tokenizer.tokenize(answer)
    max_nb_words = max( max_nb_words, len(words) )

    for phrase in [record['premise'], record['question']]:
        words = tokenizer.tokenize(phrase)
        all_words.update(words)
        max_inputseq_len = max( max_inputseq_len, len(words) )


for word in wc2v.vocab:
    all_words.add(word)

print('max_nb_words={}'.format(max_nb_words))
print('max_inputseq_len={}'.format(max_inputseq_len))

# --------------------------------------------------------------------------

#w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=32.txt'
#w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=48.txt'
#w2v_path = '/home/eek/polygon/WordSDR2/sdr.dat'
#w2v_path = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=5_DIM=128.txt'
#w2v_path = r'f:\Word2Vec\word_vectors_cbow=1_win=5_dim=32.txt'
w2v_path = os.path.join(tmp_folder,'w2v.CBOW=1_WIN=5_DIM=8.txt')
print( 'Loading the w2v model {}'.format(w2v_path) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('txt'))
w2v_dims = len(w2v.syn0[0])
print('w2v_dims={0}'.format(w2v_dims))

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
# -------------------------------------------------------------------


print('Constructing the NN model...')

nb_filters = 128
rnn_size = word_dims

final_merge_size = 0

# --------------------------------------------------------------------------------

if RUN_MODE == 'train':

    print('Building the NN computational graph: {}'.format(NET_ARCH))

    arch_filepath = os.path.join(tmp_folder, 'nn_words_generator.arch')
    log_filename = os.path.join(tmp_folder, 'nn_words_generator.log')

    if os.path.exists(log_filename):
        os.remove(log_filename)

    with open(log_filename, 'a') as wrt:
        wrt.write('NET_ARCH={} batch_size={}\n'.format(NET_ARCH, batch_size))

    words_net1 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words1')
    words_net2 = Input(shape=(max_inputseq_len, word_dims,), dtype='float32', name='input_words2')

    conv1 = []
    conv2 = []
    encoder_size = 0

    if NET_ARCH == 'lstm(cnn)':
        for kernel_size in range(1, 4):
            # сначала идут сверточные слои, образующие детекторы словосочетаний
            # и синтаксических конструкций
            conv = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          strides=1,
                          name='shared_conv_{}'.format(kernel_size))

            lstm = recurrent.LSTM(rnn_size, return_sequences=False)
            pooler = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')

            conv_layer1 = conv(words_net1)
            conv_layer1 = pooler(conv_layer1)
            conv_layer1 = lstm(conv_layer1)
            conv1.append(conv_layer1)

            conv_layer2 = conv(words_net2)
            conv_layer2 = pooler(conv_layer2)
            conv_layer2 = lstm(conv_layer2)
            conv2.append(conv_layer2)

            encoder_size += rnn_size

    encoder_merged = keras.layers.concatenate(inputs=list(itertools.chain(conv1, conv2)))
    encoder_final = Dense(units=encoder_size, activation='relu')(encoder_merged)

    # декодер должен создать вектор слова ответа по модели wc2v

    decoder = Dense(units=encoder_size, activation='relu')(encoder_final)
    decoder = Dense(units=encoder_size, activation='relu')(decoder)
    decoder = Dense(units=wc2v_dims, activation='relu', name='output')(decoder)
    #decoder = Dense(units=wc2v_dims, activation='sigmoid', name='output')(decoder)

    model = Model(inputs=[words_net1, words_net2], outputs=decoder)
    model.compile(loss='mse', optimizer='nadam')

    #model.summary()

    with open(arch_filepath, 'w') as f:
        f.write(model.to_json())

    keras.utils.plot_model(model, to_file=os.path.join(tmp_folder, 'nn_words_generator.png'), show_shapes=False, show_layer_names=True)

    trained_itargets = []

    for itarget in range(0, 1):  #max_nb_words):
        weights_path = os.path.join(tmp_folder, 'nn_words_generator.itarget={}.weights'.format(itarget))

        input_data = []
        output_data = []

        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Extracting phrases'):
            premise = row['premise']
            question = row['question']
            answer = row['answer']

            premise_words = rpad_wordseq( tokenizer.tokenize(premise), max_inputseq_len )
            question_words = rpad_wordseq( tokenizer.tokenize(question), max_inputseq_len )
            answer_words = tokenizer.tokenize(answer)
            if len(answer_words) > itarget:
                answer_word = answer_words[itarget]
                if answer_word in wc2v:
                    output_data.append(answer_word)
                    input_data.append((premise_words, question_words, premise, question))

        if len(input_data) > 10*batch_size:
            SEED = 123456
            TEST_SHARE = 0.2
            train_input, val_input, train_output, val_output = train_test_split(input_data,
                                                                                output_data,
                                                                                test_size=TEST_SHARE,
                                                                                random_state=SEED)

            nb_train_patterns = len(train_input)
            nb_valid_patterns = len(val_input)

            print('Start training for itarget={} using {} patterns for training, {} for validation...'.format(itarget, nb_train_patterns, nb_valid_patterns))

            monitor_metric = 'val_loss'

            #model_checkpoint = ModelCheckpoint(weights_path, monitor=monitor_metric,
            #                                   verbose=1, save_best_only=True, mode='auto')
            #early_stopping = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='auto')

            viz = VisualizeCallback(val_input, val_output, model, wc2v, weights_path, itarget, batch_size)

            callbacks = [viz]

            hist = model.fit_generator(generator=generate_rows(train_input, train_output, batch_size, wc2v, 1),
                                       steps_per_epoch=int(nb_train_patterns/batch_size),
                                       epochs=200,
                                       verbose=1,
                                       callbacks=callbacks,
                                       validation_data=generate_rows(val_input, val_output, batch_size, wc2v, 1),
                                       validation_steps=int(nb_valid_patterns/batch_size)
                                       )
            trained_itargets.append(itarget)

            with open(log_filename, 'a') as wrt:
                wrt.write('itarget={} max_accuracy={}\n'.format(itarget, viz.best_val_acc))

    # сохраним конфиг модели, чтобы ее использовать в чат-боте
    model_config = {
                    'max_inputseq_len': max_inputseq_len,
                    'max_nb_words': max_nb_words,
                    'w2v_path': w2v_path,
                    'wc2v_dims': wc2v_dims,
                    'wordchar2vector_path': wordchar2vector_path,
                    'PAD_WORD': PAD_WORD,
                    'model_folder': tmp_folder,
                    'word_dims': word_dims,
                    'trained_itargets': trained_itargets,
                   }

    with open(os.path.join(tmp_folder,'nn_words_generator.config'), 'w') as f:
        json.dump(model_config, f)


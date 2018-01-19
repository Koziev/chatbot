# -*- coding: utf-8 -*-
'''
Тренировка модели для превращения символьной цепочки слова в вектор.
RNN и CNN варианты энкодера.
Декодер - LSTM.
(c) by Koziev Ilya inkoziev@gmail.com
'''

from __future__ import print_function
from keras.layers.core import Activation, RepeatVector, Dense, Masking
from keras.layers.wrappers import TimeDistributed
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Dropout, Input, Permute, Flatten, Reshape
from keras.layers import concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import TimeDistributed
from keras.models import Model
from keras.callbacks import CSVLogger
import numpy as np
import os
import codecs
import itertools
import pickle
import re
import zipfile
import random
from Tokenizer import Tokenizer
import pandas as pd


tmp_folder = '../tmp'
data_folder = '../data'


ARCH_TYPE = 'rnn' # архитектура модели: cnn - сверточный энкодер, rnn - рекуррентный энкодер
vec_size = 56 # размер вектора представления слова
n_misspelling_per_word = 0 # кол-во добавляемых вариантов с опечатками на одно исходное слово
tunable_char_embeddings = False # делать ли настраиваемые векторы символов (True) или 1-hot (False)
batch_size = 250


# Из этого текстового файла возьмем слова, на которых будем тренировать модель встраивания.
# corpus_path = r'f:\Corpus\word2vector\ru\SENTx.corpus.w2v.txt'
corpus_path = '/home/eek/Corpus/word2vector/ru/SENTx.corpus.w2v.txt'
#paraphrases_path = '../data/paraphrases.csv'
paraphrases_path = '../data/premise_question_relevancy.csv'
pqa_path = '../data/premise_question_answer.csv'

FILLER_CHAR =u' '
BEG_CHAR = u'['
END_CHAR = u']'


TRAIN = True

while True:
    a1 = raw_input('0-train model\n1-calculate embeddings using pretrained model\n[0/1]: ')
    if a1=='0':
        TRAIN = True
        print('Train model')
        break
    elif a1=='1':
        TRAIN = False
        print('Calculate embeddings')
        break
    else:
        print('Unrecognized choice "{}"'.format(a1))

# ---------------------------------------------------------------




char_replacement = {}
cx = u'оашщеиьъдтсзбпхк'
for i in range(0, len(cx), 2):
    char_replacement[cx[i]] = cx[i + 1]

pattern_replacement = {}
pattern_replacement[u'тся'] = u'ться'
pattern_replacement[u'ться'] = u'тся'
pattern_replacement[u'жи'] = u'жы'
pattern_replacement[u'жы'] = u'жи'
pattern_replacement[u'ши'] = u'шы'
pattern_replacement[u'шы'] = u'ши'
pattern_replacement[u'ца'] = u'тся'
pattern_replacement[u'ца'] = u'тса'
pattern_replacement[u'тса'] = u'ца'
pattern_replacement[u'ча'] = u'чя'
pattern_replacement[u'ща'] = u'щя'
pattern_replacement[u'сч'] = u'щ'
pattern_replacement[u'съе'] = u'се'
pattern_replacement[u'съе'] = u'се'
pattern_replacement[u'въе'] = u'ве'
pattern_replacement[u'ве'] = u'въе'
pattern_replacement[u'ого'] = u'ова'
pattern_replacement[u'стн'] = u'сн'
pattern_replacement[u'нн'] = u'н'
pattern_replacement[u'сс'] = u'с'
pattern_replacement[u'рр'] = u'р'
pattern_replacement[u'дт'] = u'тт'


def pad_word( word, max_word_len ):
    return BEG_CHAR + word + END_CHAR + (max_word_len-len(word))*FILLER_CHAR


def unpad_word(word):
    return word.strip()[1:-1]


def augment_wordset( wordset, known_words, max_word_len ):
    reslist = set()
    for word in wordset:
        reslist.add( (pad_word(word, max_word_len), pad_word(word, max_word_len)) )
        wlen = len(word)
        for _ in range(n_misspelling_per_word):
            word2 = u''
            scenario = random.randint(0, 4)

            if scenario == 0:
                # удваиваем любую букву
                ichar = random.randint(0, wlen - 1)
                ch = word[ichar]
                if ichar == 0:
                    word2 = ch + word  # удваиваем первый символ
                elif ichar == wlen - 1:
                    word2 = word + ch  # удваиваем последний символ
                else:
                    word2 = word[:ichar + 1] + ch + word[ichar + 1:]  # удваиваем символ внутри слова

            elif scenario == 1:
                # удаляем любую букву
                ichar = random.randint(0, wlen - 1)
                if ichar == 0:
                    word2 = word[1:]  # удаляем первый символ
                elif ichar == wlen - 1:
                    word2 = word[:max_word_len - 1]  # удаляем последний символ
                else:
                    word2 = word[:ichar] + word[ichar + 1:]  # удаляем символ внутри слова

            elif scenario == 2:
                # замены букв
                replacement_count = 0
                for ch in word:

                    ch2 = ch

                    if replacement_count == 0:
                        if ch in char_replacement:
                            ch2 = char_replacement[ch]

                    if ch != ch2:
                        replacement_count += 1

                    word2 += ch2

            elif scenario == 3:
                # сложные замены цепочек букв типа ТСЯ-ТЬСЯ
                for (seq1, seq2) in pattern_replacement.items():
                    if word.find(seq1) != -1:
                        word2 = word.replace(seq1, seq2)
                        break

            if word!=word2 and word2 != '' and word2 not in known_words and len(word2) <= max_word_len:
                reslist.add(( pad_word(word, max_word_len), pad_word(word2, max_word_len)))

    return reslist


def raw_wordset( wordset, max_word_len ):
    return [ (pad_word(word, max_word_len),pad_word(word, max_word_len)) for word in wordset ]


rx1 = re.compile( u'[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]+' )
dict_words = set()
with zipfile.ZipFile( os.path.join(data_folder,'ruwords.txt.zip')) as z:
    with z.open('ruwords.txt') as rdr:
        for line in rdr:
            word = line.decode('utf-8').strip()
            if rx1.match( word) is not None:
                dict_words.add(word)

known_words = set()
with codecs.open(corpus_path, 'r', 'utf-8') as rdr:
    line_count = 0
    for line0 in rdr:
        #line = line0.decode('utf-8').strip()
        line = line0.strip()
        words = line.split(u' ')
        known_words.update(words)
        line_count += 1
        if line_count>1000000:
            break

df = pd.read_csv(paraphrases_path, encoding='utf-8', delimiter='\t', quoting=3)
tokenizer = Tokenizer()
#for phrase in itertools.chain(df['phrase1'].values, df['phrase2'].values):
for phrase in itertools.chain(df['premise'].values, df['question'].values):
    words = tokenizer.tokenize(phrase.lower())
    known_words.update(words)

#known_words &= dict_words  # если хотим тренироваться только по верифицированным словарным лексемам

print('There are {} known words'.format(len(known_words)))

max_word_len = max( map(len,known_words) )
seq_len = max_word_len+2 # 2 символа добавляются к каждому слову для маркировки начала и конца последовательности
print('max_word_len={}'.format(max_word_len))

val_share = 0.3
train_words = set( filter( lambda z:random.random()>val_share, known_words ) )
val_words = set( filter( lambda z:z not in train_words, known_words) )

train_words = augment_wordset(train_words, known_words, max_word_len)
#val_words = augment_wordset(val_words, known_words, max_word_len)
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

    def __init__(self, X_test, y_test, model, index2char):
        self.epoch = 0
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.index2char = index2char
        self.output_path = os.path.join( tmp_folder, 'chars2vector.results.txt' )
        if os.path.isfile(self.output_path):
            os.remove(self.output_path)


    def decode_char_indeces(self, char_indeces):
        return u''.join([ self.index2char[c] for c in char_indeces ]).strip()


    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch+1
        with open( self.output_path, 'a' ) as fsamples:
            fsamples.write( u'\n' + '='*50 + u'\nepoch=' + str(self.epoch) + u'\n' );
            for i in range(10):
                ind = np.random.randint(0, len(self.X_test))
                rowX, rowy = self.X_test[np.array([ind])], self.y_test[np.array([ind])]
                preds = self.model.predict(rowX, verbose=0)

                correct = self.decode_char_indeces(rowy[0,:,:].argmax(axis=-1))
                #guess = self.ctable.decode(preds[0], calc_argmax=False)

                predicted_char_indeces = preds[0,:,:].argmax(axis=-1)
                guess = self.decode_char_indeces(predicted_char_indeces)

                if i<10:
                    print(colors.ok + '☑ ' + colors.close if correct == guess else colors.fail + '☒ ' + colors.close, end='')
                    print(u'wordform={} model_output={}'.format(correct, guess) )

                fsamples.write( (correct + u' ==> ' + guess + u'\n').encode('utf-8') )

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

weigths_path = os.path.join(tmp_folder, 'wordcharsvector.model')

model_checkpoint = ModelCheckpoint( weigths_path,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto')

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=20,
                               verbose=1,
                               mode='auto')


X_viz, y_viz = build_test( list(val_words)[0:1000], max_word_len, char2index )

visualizer = VisualizeCallback(X_viz, y_viz, model, index2char)

learning_curve_filename = os.path.join( tmp_folder, 'learning_curve_{}_n_misspell={}_vecsize={}_tunable_char_embeddings={}.csv'.format(ARCH_TYPE, n_misspelling_per_word, vec_size, tunable_char_embeddings) )
csv_logger = CSVLogger(learning_curve_filename, append=True, separator='\t')


if TRAIN:
    hist = model.fit_generator(generator=generate_rows(train_words, batch_size, char2index, 1),
                               steps_per_epoch=int(len(train_words)/batch_size),
                               epochs=100,
                               verbose=1,
                               callbacks=[model_checkpoint, early_stopping, visualizer, csv_logger],
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
with open(os.path.join(tmp_folder, 'wordchar2vector.arch'), 'w') as f:
    f.write(model.to_json())

if TRAIN:
    model.save_weights(weigths_path)
else:
    model.load_weights(weigths_path)

output_words = set(known_words)
tokenizer = Tokenizer()
df = pd.read_csv(paraphrases_path, encoding='utf-8', delimiter='\t', quoting=3)
for phrase in itertools.chain( df['premise'].values, df['question'].values ):
    words = tokenizer.tokenize(phrase)
    output_words.update(words)

df = pd.read_csv(pqa_path, encoding='utf-8', delimiter='\t', quoting=3)
for phrase in itertools.chain( df['premise'].values, df['question'].values, df['answer'].values ):
    words = tokenizer.tokenize(phrase)
    output_words.update(words)



nb_words = len(output_words)
with codecs.open( os.path.join( tmp_folder, 'wordchar2vector.dat'), 'w', 'utf-8') as wrt:
    wrt.write('{} {}\n'.format(nb_words, vec_size))

    nb_batch = int(nb_words/batch_size) + (0 if (nb_words%batch_size)==0 else 1)
    wx = list(output_words)
    #dummy = wx.index(u'лаконичная')
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
            #if u'мышка' in naked_word:
            #    print(naked_word)
            wrt.write(u'{} {}\n'.format(naked_word, u' '.join([str(x) for x in word_vect]) ))

        word_index += nw
        words_remainder -= nw

print('\nDone.')

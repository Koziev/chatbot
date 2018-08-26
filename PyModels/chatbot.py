# -*- coding: utf-8 -*-
'''
*** УДАЛИТЬ ***

Базовая модель чатбота, использующая ранее обученные модели для сопоставления предложений
и производства логических выводо
'''

from __future__ import division  # for python2 compatability
from __future__ import print_function

import itertools
import json
import os
import sys

import gensim
import numpy as np
from keras.models import model_from_json

from utils.tokenizer import Tokenizer

input_path = '../data/paraphrases.csv'
tmp_folder = '../tmp'
data_folder = '../data'



# -------------------------------------------------------------------------

model_config = None
with open(os.path.join(tmp_folder,'rnn_detector.config'), 'r') as f:
    model_config = json.load(f)

max_wordseq_len = model_config['max_wordseq_len']
w2v_path = model_config['w2v_path']
wordchar2vector_path = model_config['wordchar2vector_path']
PAD_WORD = model_config['PAD_WORD']
arch_filepath = model_config['arch_filepath']
weights_path = model_config['weights_path']
word_dims = model_config['word_dims']

# --------------------------------------------------------------------------

with open(arch_filepath, 'r') as f:
    model1 = model_from_json(f.read())

model1.load_weights(weights_path)


# --------------------------------------------------------------------------

print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])
print('wc2v_dims={0}'.format(wc2v_dims))

# --------------------------------------------------------------------------

print( 'Loading the w2v model {}'.format(w2v_path) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
w2v_dims = len(w2v.syn0[0])
print('w2v_dims={0}'.format(w2v_dims))

# --------------------------------------------------------------------------

def pad_wordseq(words, n):
    return list( itertools.chain( itertools.repeat(PAD_WORD, n-len(words)), words, ) )



def vectorize_words( words, X_batch, irow, w2v, wc2v ):
    for iword,word in enumerate( words ):
        if word in w2v:
            X_batch[irow, iword, :w2v_dims] = w2v[word]
        if word in wc2v:
            X_batch[irow, iword, w2v_dims:] = wc2v[word]


# ---------------------------------------------------------------------------

memory_phrases = []
tokenizer = Tokenizer()

trace_enabled = False

while True:
    phrase = raw_input(':> ').decode(sys.stdout.encoding).strip().lower()
    if len(phrase)==0:
        break

    # проверим, что все слова в этой фразе нам известны (хотя в будущем
    # надо просто использовать тут wc2v модель, которая из любой символьной
    # цепочки сделает вектор и проблемы не будет).
    words = tokenizer.tokenize(phrase)
    all_words_known = True

    for word in itertools.chain(words):
        if word not in wc2v:
            print(u'Unknown word {}'.format(word))
            all_words_known = False


    if all_words_known:
        if phrase[-1]==u'?':

            nb_answers = len(memory_phrases)
            X1_probe = np.zeros((nb_answers, max_wordseq_len, word_dims), dtype=np.float32)
            X2_probe = np.zeros((nb_answers, max_wordseq_len, word_dims), dtype=np.float32)

            words1 = tokenizer.tokenize(phrase)

            best_answer = ''
            best_sim = 0.0
            for ianswer,answer in enumerate(memory_phrases):
                words2 = tokenizer.tokenize(answer)

                vectorize_words(pad_wordseq(words1, max_wordseq_len), X2_probe, ianswer, w2v, wc2v)
                vectorize_words(pad_wordseq(words2, max_wordseq_len), X1_probe, ianswer, w2v, wc2v)

                if False:
                    # содержимое X*_probe для отладки
                    with open('../tmp/X_probe.chatbot.txt', 'w') as wrt:
                        for X, name in [(X1_probe,'X1_probe'), (X2_probe,'X2_probe')]:
                            wrt.write('{}\n'.format(name))
                            for i in range(X.shape[1]):
                                for j in range(X.shape[2]):
                                    wrt.write(' {}'.format(X[0,i,j]))
                                wrt.write('\n')
                        wrt.write('\n\n')

            y_probe = model1.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})
            for ianswer,answer in enumerate(memory_phrases):
                sim = y_probe[ianswer]
                if sim>best_sim:
                    best_sim = sim
                    best_answer = answer

            if trace_enabled:
                print(u'{} ==> {}'.format(best_sim, best_answer))
            else:
                if best_sim>0.5:
                    print(u'да')
                else:
                    print(u'нет')

        else:
            if phrase not in memory_phrases:
                    memory_phrases.append(phrase)

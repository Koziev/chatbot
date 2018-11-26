# -*- coding: utf-8 -*-
'''
Тренировка модели частеречной разметки (part-of-speech tagging) для русского языка.
Используется CRFSuite и фичи на базе w2v+wc2v
Для чатбота https://github.com/Koziev/chatbot.
'''

from __future__ import print_function
from __future__ import division  # for python2 compatibility

import codecs
import os
import gc
import itertools
import logging
import numpy as np
import platform
import logging.handlers
import gensim
import json

import utils.logging_helpers


winspan = 1  # для каждого токена фичи берутся с окна +/- winspan; 0 - только сам токен.
C1 = 0.80  # coefficient for L1 penalty in CRFSuite
C2 = 5e-3  # coefficient for L2 penalty in CRFSuite
nb_iters = 100  # макс. число итераций обучения для CRFSuite
max_nb_instances = 100000  # кол-во предложений для обучения (ограничение датасета для ускорения отладки)
use_gren = True  # использовать ли грамматический словарь как доп. источник фич для слов

data_folder = '../../data'
corpora = ['united_corpora.dat', 'morpheval_corpus_solarix.full.dat']
crftrain_path = '../../tmp/postagger_train.dat'
crftest_path = '../../tmp/postagger_val.dat'
model_path = '../../tmp/postagger.model'
tmp_folder = '../../tmp'

gren_path = '../../data/word2tags.dat'
wordchar2vector_path = '../../data/wordchar2vector.dat'
word2vector_path = os.path.expanduser('~/polygon/w2v/w2v.CBOW=1_WIN=5_DIM=64.bin')

BEG_TOKEN = '<beg>'
END_TOKEN = '<end>'


def normalize_word(word):
    return word.replace(' - ', '-').replace(u'ё', u'е').lower()


def build_label(part_of_speech, tags):
    label = part_of_speech
    if tags != '_':
        # для важных частей речи берем все теги, для прочих остается только
        # наименование части речи.
        if part_of_speech in ['NOUN', 'ADJ', 'VERB']:
            label = label + '|' + tags
    return label


def get_word_features(word, prefix, word2vec, word2tags):
    features = []

    if use_gren and word in word2tags:
        for tag in word2tags[word]:
            features.append((u'tag[{}]={}'.format(prefix, tag.replace(':', '=')), 1.0))

    if word in word2vec:
        v = word2vec[word]
        for i, x in enumerate(v):
            if x > 0.0:
                features.append(('0_{}[{}]'.format(i, prefix), x))
            elif x < 0.0:
                features.append(('1_{}[{}]'.format(i, prefix), -x))

    return features


def vectorize_sample(lines, word2vec, word2tags):
    lines1 = []
    for line in lines:
        word = line[0]
        if word not in [BEG_TOKEN, END_TOKEN]:
            if word not in word2vec:
                print(u'DEBUG @76: word {} missing'.format(word))
                return None

        pos = line[1]
        tags = line[2]
        label = build_label(pos, tags)
        lines1.append((word, label))

    lines2 = []
    nb_words = len(lines1)
    for iword, data0 in enumerate(lines1):
        label = data0[1]
        word_features = dict()
        for j in range(-winspan, winspan+1):
            iword2 = iword + j
            if nb_words > iword2 >= 0:
                data = lines1[iword2]
                features = get_word_features(data[0], str(j), word2vec, word2tags)
                word_features.update(features)

        lines2.append((word_features, label))

    return lines2


def print_word_features(features, separator):
    return separator.join(u'{}:{}'.format(f, v) for (f, v) in features)
    #return separator.join('{}'.format(f, v) for (f, v) in features)



# настраиваем логирование в файл
utils.logging_helpers.init_trainer_logging(os.path.join(tmp_folder, 'train_postagger.log'))

sample_count = 0

# Соберем лексикон
all_words = set()
logging.info(u'Lexicon building...')
for corpus in corpora:
    sent_lines = []
    with codecs.open(os.path.join(data_folder, corpus), 'r', 'utf-8') as rdr:
        for iline, line in enumerate(rdr):
            if ((1+iline) % 1000000) == 0:
                logging.info(u'{} lines in {} processed'.format(iline, corpus))
            line = line.strip()
            if line:
                tx = line.split('\t')
                word = normalize_word(tx[1])
                all_words.add(word)

logging.info('{} words in lexicon'.format(len(all_words)))

# Загружаем грамматический словарь
word2tags = {BEG_TOKEN: [BEG_TOKEN], END_TOKEN: [END_TOKEN]}
if use_gren:
    logging.info(u'Loading grammar dictionary from {}...'.format(gren_path))
    with codecs.open(gren_path, 'r', 'utf-8') as rdr:
        for line in rdr:
            tx = line.strip().split('\t')
            if len(tx) == 4:
                word = normalize_word(tx[0])
                if word in all_words:
                    pos = tx[1]
                    tags = tx[3].split(' ')
                    tags = set(itertools.chain([pos], tags))
                    if word not in word2tags:
                        word2tags[word] = tags
                    else:
                        word2tags[word].update(tags)


logging.info(u'Loading the wordchar2vector model {}'.format(wordchar2vector_path))
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])
logging.info('wc2v_dims={0}'.format(wc2v_dims))

logging.info(u'Loading the w2v model {}'.format(word2vector_path))
w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
w2v_dims = len(w2v.syn0[0])
logging.info('w2v_dims={0}'.format(w2v_dims))

word_dims = w2v_dims + wc2v_dims

word2vec = dict()
for word in all_words:
    if word in wc2v:
        v = np.zeros(word_dims)
        v[w2v_dims:] = wc2v[word]
        if word in w2v:
            v[:w2v_dims] = w2v[word]

        word2vec[word] = v

del w2v
del wc2v
gc.collect()

wrt_train = codecs.open(crftrain_path, 'w', 'utf-8')
wrt_test = codecs.open(crftest_path, 'w', 'utf-8')

add_sent_walls = True

logging.info('Vectorization of samples...')
sample_count = 0
nb_train = 0
nb_test = 0
all_labels = set()
val_samples = []
for corpus in corpora:
    logging.info(u'Start processing {}'.format(corpus))
    sent_lines = []
    with codecs.open(os.path.join(data_folder, corpus), 'r', 'utf-8') as rdr:
        for iline, line in enumerate(rdr):
            if sample_count >= max_nb_instances:
                break

            if ((1+iline) % 100000) == 0:
                logging.info('{} lines processed'.format(sample_count))
                #break  # DEBUG!!!

            line = line.strip()
            if line:
                if len(sent_lines) == 0 and add_sent_walls:
                    sent_lines.append((BEG_TOKEN, BEG_TOKEN, ''))

                tx = line.split('\t')
                word = normalize_word(tx[1])
                pos = tx[3]
                tags = tx[4]
                sent_lines.append((word, pos, tags))
            else:
                sample_count += 1
                if add_sent_walls:
                    sent_lines.append((END_TOKEN, END_TOKEN, ''))
                lines2 = vectorize_sample(sent_lines, word2vec, word2tags)
                if lines2 is not None:

                    # начало отладки
                    #for line2 in lines2:
                    #    print(u'{}\t{}'.format(print_word_features(line2[0]), line2[1]))
                    #exit(0)
                    # конец отладки

                    if (sample_count%5) == 0:
                        wrt = wrt_test
                        nb_test += 1

                        val_sample = [(word, build_label(pos, tags)) for (word, pos, tags) in sent_lines]
                        val_samples.append(val_sample)
                    else:
                        wrt = wrt_train
                        nb_train += 1

                    for line2 in lines2:
                        label = line2[1]
                        all_labels.add(label)
                        wrt.write(u'{}\t{}\n'.format(label, print_word_features(line2[0], '\t')))
                    wrt.write('\n')

                sent_lines = []

wrt_train.close()
wrt_test.close()

logging.info('{} instances processed'.format(sample_count))
logging.info('all_labels.count={}'.format(len(all_labels)))

# сохраним конфиг модели
model_config = {'w2v_path': word2vector_path,
                'wc2v_path': wordchar2vector_path,
                'winspan': winspan,
                'use_gren': use_gren,
                'model_path': model_path
                }
with open(os.path.join('../../tmp', 'postagger.config'), 'w') as wrt:
    json.dump(model_config, wrt)


# запуск обучения через консольный скрипт
if platform.system() == 'Linux':
    crfsuite_exe = 'crfsuite'
else:
    crfsuite_exe = r'..\..\crf\CRFSuite.exe'


train_cmd = u'{0} learn --set=c1={1} --set=c2={2} --set=max_iterations={3} -m {4} {5}'.format(crfsuite_exe,
                                                                                              C1, C2,
                                                                                              nb_iters,
                                                                                              model_path,
                                                                                              crftrain_path)
logging.info(u'Start training CRF model on {} instances in {} using shell command:\n{}'.format(nb_train, crftrain_path, train_cmd))
s = os.system(train_cmd)
logging.info('Done, exit status={}'.format(s))

# оценка качества
logging.info(u'Start validating CRF model on {} instances in {}'.format(nb_test, crftest_path))
tagging_path = os.path.join(tmp_folder, 'crf_tagging.txt')
s = os.system('{} tag -r -m {} {} > {}'.format(crfsuite_exe, model_path, crftest_path, tagging_path))
logging.info('Done, exit status={}'.format(s))

nb_instance = 0
nb_error = 0

val_results_path = '../../tmp/postagger.crf_val_results.txt'
logging.info(u'Writing validation results to {}'.format(val_results_path))
test_counter = 0
token_errors = 0
tokens_count = 0
itoken = 0
with codecs.open(val_results_path, 'w', 'utf-8') as wrt_val_results:
    with open(tagging_path, 'r') as rdr:
        instance_has_error = 0
        for line in rdr:
            if len(line) == 0:
                break

            line = line.strip()
            if len(line) == 0:
                nb_instance += 1
                itoken = 0

                if instance_has_error:
                    nb_error += 1

                test_counter += 1
                instance_has_error = 0
                wrt_val_results.write(u'\n\n'.format(val_word, val_label, pred_label))
            else:
                tx = line.split('\t')

                val_token = val_samples[nb_instance][itoken]
                val_word = val_token[0]
                val_label = val_token[1]
                pred_label = tx[1]
                if tx[0] != val_label:
                    logging.error(u'ERROR@296: tx[0]={} val_label={}'.format(tx[0], val_label))

                if tx[0] not in [BEG_TOKEN, END_TOKEN]:
                    tokens_count += 1
                    if tx[0] != tx[1]:
                        instance_has_error += 1
                        token_errors += 1
                        wrt_val_results.write(u'{:20s} true={:50s} pred={} <-- ERROR\n'.format(val_word, val_label, pred_label))
                    else:
                        wrt_val_results.write(u'{:20s} true={:50s} pred={}\n'.format(val_word, val_label, pred_label))
                itoken += 1

accuracy = 1.0 - float(token_errors) / float(tokens_count)
logging.info(u'Per token accuracy={}'.format(accuracy))

accuracy = 1.0 - float(nb_error) / float(nb_instance)
logging.info(u'Per instance accuracy={}'.format(accuracy))

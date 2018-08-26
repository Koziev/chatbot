# -*- coding: utf-8 -*-
'''
Ответчик.
Использует все ранее обученные скриптами relevancy_model и qa_model модели
и выполняет следующие действия:
1) с консоли вводится вопрос.
2) ищется наиболее подходящая предпосылка из списка, загружаемого из текстового файла.
3) определяется способ конструирования ответа - yes/no или word copy.
4) ответ генерируется и печатается.
'''

from __future__ import division  # for python2 compatability
from __future__ import print_function

import codecs
import itertools
import json
import os
import pickle
import platform
import re
import sys
from datetime import datetime

import colorama  # https://pypi.python.org/pypi/colorama
import gensim
import numpy as np
import xgboost
from keras.models import model_from_json
from scipy.sparse import lil_matrix

from utils.tokenizer import Tokenizer

input_path = '../data/paraphrases.csv'
tmp_folder = '../tmp'
data_folder = '../data'
premises_path = '../data/premises.txt'
premises_1s_path = '../data/premises_1s.txt'
premises_2s_path = '../data/premises_2s.txt'



# -------------------------------------------------------------------

BEG_WORD = '\b'
END_WORD = '\n'


def canonize_text(s):
    s = re.sub("(\\s{2,})", ' ', s.strip())
    return s


def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    return u' '.join(itertools.chain([BEG_WORD], filter( lambda z:len(z)>0, words), [END_WORD]))


# --------------------------------------------------------------------------


def xgb_yesno_vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, shingle2id):
    ps = set(premise_shingles)
    qs = set(question_shingles)
    common_shingles = ps & qs
    notmatched_ps = ps - qs
    notmatched_qs = qs - ps

    nb_shingles = len(shingle2id)

    icol = 0
    for shingle in common_shingles:
        if shingle not in shingle2id:
            print(u'Missing shingle {} in yes_no data'.format(shingle))
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_ps:
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_qs:
        X_data[idata, icol+shingle2id[shingle]] = True



# -------------------------------------------------------------------------

def xgb_relevancy_vectorize_sample_x(X_data, idata, premise_shingles, question_shingles, shingle2id):
    ps = set(premise_shingles)
    qs = set(question_shingles)
    common_shingles = ps & qs
    notmatched_ps = ps - qs
    notmatched_qs = qs - ps

    nb_shingles = len(shingle2id)

    icol = 0
    for shingle in common_shingles:
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_ps:
        X_data[idata, icol+shingle2id[shingle]] = True

    icol += nb_shingles
    for shingle in notmatched_qs:
        X_data[idata, icol+shingle2id[shingle]] = True


# -------------------------------------------------------------------------

trace_enabled = False

# -------------------------------------------------------------------------

# Общие параметры для сеточных моделей
with open(os.path.join(tmp_folder,'qa_model.config'), 'r') as f:
    model_config = json.load(f)

    max_inputseq_len = model_config['max_inputseq_len']
    max_outputseq_len = model_config['max_outputseq_len']
    w2v_path = model_config['w2v_path']
    wordchar2vector_path = model_config['wordchar2vector_path']
    PAD_WORD = model_config['PAD_WORD']
    word_dims = model_config['word_dims']


# -------------------------------------------------------------------------

if trace_enabled:
    print('Loading the relevancy model...')

if False:
    # Сеточная модель для определения релевантности вопроса и предпосылки
    model_config = None

    with open(os.path.join(tmp_folder, 'relevancy_model.config'), 'r') as f:
        model_config = json.load(f)

    #max_wordseq_len = model_config['max_wordseq_len']
    #w2v_path = model_config['w2v_path']
    #wordchar2vector_path = model_config['wordchar2vector_path']
    #PAD_WORD = model_config['PAD_WORD']
    arch_filepath = model_config['arch_filepath']
    weights_path = model_config['weights_path']
    #word_dims = model_config['word_dims']

    with open(arch_filepath, 'r') as f:
        relevancy_model = model_from_json(f.read())

    relevancy_model.load_weights(weights_path)
else:
    # Определение релевантности предпосылки и вопроса на основе XGB модели
    with open(os.path.join(tmp_folder, 'xgb_relevancy.config'), 'r') as f:
        model_config = json.load(f)

    xgb_relevancy_shingle2id = model_config['shingle2id']
    xgb_relevancy_shingle_len = model_config['shingle_len']
    xgb_relevancy_nb_features = model_config['nb_features']

    # 'model_filename': model_filename,
    xgb_relevancy = xgboost.Booster()
    xgb_relevancy.load_model(model_config['model_filename'])



# --------------------------------------------------------------------------


# Модель для выбора ответов yes|no на базе XGB
with open(os.path.join(tmp_folder,'xgb_yes_no.config'), 'r') as f:
    model_config = json.load(f)

xgb_yesno_shingle2id = model_config['shingle2id']
xgb_yesno_shingle_len = model_config['shingle_len']
xgb_yesno_nb_features = model_config['nb_features']
xgb_yesno_feature_names = model_config['feature_names']

xgb_yesno = xgboost.Booster()
xgb_yesno.load_model(model_config['model_filename'])

# --------------------------------------------------------------------------


models = dict()
#for model_label in ['model_selector', 'yes_no', 'word_copy']:
for model_label in ['model_selector', 'word_copy3']:
    if trace_enabled:
        print('Loading "{}" model...'.format(model_label))

    arch_filepath = os.path.join(tmp_folder, 'qa_{}.arch'.format(model_label))
    weights_path = os.path.join(tmp_folder, 'qa_{}.weights'.format(model_label))
    with open(arch_filepath, 'r') as f:
        m = model_from_json(f.read())

    m.load_weights(weights_path)
    models[model_label] = m



with open(os.path.join(tmp_folder,'qa_model.config'), 'r') as f:
    qa_model_config = json.load(f)



# --------------------------------------------------------------------------

# Классификатор грамматического лица на базе XGB
config_path = os.path.join(tmp_folder, 'xgb_person_classifier.config')
with open(config_path,'r') as f:
    person_classifier_config = json.load(f)

xgb_person_classifier_shingle_len = person_classifier_config['shingle_len']
xgb_person_classifier_shingle2id = person_classifier_config['shingle2id']
xgb_person_classifier_nb_features = person_classifier_config['nb_features']

xgb_person_classifier = xgboost.Booster()
xgb_person_classifier.load_model(person_classifier_config['model_filename'])

# ------------------------------------------------------------------------------

# Нейросетевые модели для манипуляции с грамматическим лицом

#for model_label in ['person_classifier', 'changeable_word']:
for model_label in ['changeable_word']:
    if trace_enabled:
        print('Loading "{}" model...'.format(model_label))

    arch_filepath = os.path.join(tmp_folder, 'person_change_{}.arch'.format(model_label))
    weights_path = os.path.join(tmp_folder, 'person_change_{}.weights'.format(model_label))
    with open(arch_filepath, 'r') as f:
        m = model_from_json(f.read())

    m.load_weights(weights_path)
    models[model_label] = m

with open(os.path.join(tmp_folder,'person_change_model.config'), 'r') as f:
    person_change_model_config = json.load(f)

# --------------------------------------------------------------------------

# Упрощенная модель для работы с грамматическим лицом
with open('../tmp/person_change_dictionary.pickle', 'r') as f:
    model = pickle.load(f)

w1s = model['word_1s']
w2s = model['word_2s']
person_change_1s_2s = model['person_change_1s_2s']
person_change_2s_1s = model['person_change_2s_1s']


def get_person(phrase, tokenizer):
    for word in tokenizer.tokenize(phrase):
        if word in w1s:
            return '1s'
        elif word in w2s:
            return '2s'
    return '3'


def change_person(phrase, tokenizer, target_person):
    inwords = tokenizer.tokenize(phrase)
    outwords = []
    for word in inwords:
        if target_person=='2s' and word in w1s:
            outwords.append(person_change_1s_2s[word])
        elif target_person=='1s' and word in w2s:
            outwords.append(person_change_2s_1s[word])
        else:
            outwords.append(word)

    return u' '.join(outwords)



# --------------------------------------------------------------------------

if trace_enabled:
    print( 'Loading the wordchar2vector model {}'.format(wordchar2vector_path) )
wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
wc2v_dims = len(wc2v.syn0[0])

# --------------------------------------------------------------------------

if trace_enabled:
    print( 'Loading the w2v model {}'.format(w2v_path) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
w2v_dims = len(w2v.syn0[0])


# --------------------------------------------------------------------------

# Слева добавляем пустые слова
def pad_wordseq(words, n):
    return list(itertools.chain(itertools.repeat(PAD_WORD, n-len(words)), words))

# Справа добавляем пустые слова
def rpad_wordseq(words, n):
    return list(itertools.chain(words, itertools.repeat(PAD_WORD, n-len(words))))

# -------------------------------------------------------------------------



def vectorize_words(words, X_batch, irow, w2v, wc2v):
    for iword,word in enumerate( words ):
        if word in w2v:
            X_batch[irow, iword, :w2v_dims] = w2v[word]
        if word in wc2v:
            X_batch[irow, iword, w2v_dims:] = wc2v[word]


# ---------------------------------------------------------------------------

memory_phrases = []
for p, ptype in [(premises_path, '3'), (premises_1s_path, '1s'), (premises_2s_path, '2s')]:
    if trace_enabled:
        print('Loading premises from {}'.format(p))
    with codecs.open(p, 'r', 'utf-8') as rdr:
        for line in rdr:
            memory_phrases.append( (canonize_text(line), ptype, '') )


# Добавляем текущие факты

# День недели
dw = [u'понедельник', u'вторник', u'среда', u'четверг',
      u'пятница', u'суббота', u'воскресенье'][datetime.today().weekday()]

memory_phrases.append( (u'сегодня '+dw, '3', 'current_date') )

# Время года
#currentSecond= datetime.now().second
#currentMinute = datetime.now().minute
#currentHour = datetime.now().hour

#currentDay = datetime.now().day
cur_month = datetime.now().month
#currentYear = datetime.now().year

season = {11:u'зима', 12:u'зима', 1:u'зима',
          2:u'весна', 3:u'весна', 4:u'весна',
          5:u'лето', 6:u'лето', 7:u'лето',
          8:u'осень', 9:u'осень', 10:u'осень'}[cur_month]
memory_phrases.append( (u'сейчас '+season, '3', 'current_season') )



if trace_enabled:
    print('{} premises loaded'.format(len(memory_phrases)))

# ---------------------------------------------------------------------------

tokenizer = Tokenizer()

print(colorama.Fore.LIGHTBLUE_EX+'Answering machine is running on '+platform.platform()+colorama.Fore.RESET)


# --------------------------------------------------------------------------------
# Стартовое приветствие.
with codecs.open('../input/smalltalk_opening.txt', 'r', 'utf-8') as rdr:
    openers = []
    for line in rdr:
        if len(line) > 2:
            openers.append(line.strip())

    print(np.random.choice(openers))

# --------------------------------------------------------------------------------


while True:
    print('\n')
    question = raw_input('Q:> ').decode(sys.stdout.encoding).strip().lower()
    if len(question) == 0:
        break

    question = canonize_text(question)
    if question == u'#traceon':
        trace_enabled = True
        continue
    if question == u'#traceoff':
        trace_enabled = False
        continue
    if question == u'#facts':
        for fact,person,fact_id in memory_phrases:
            print(u'{}'.format(fact))
        continue
    if question == u'#exit':
        break

    question0 = question

    # обновляем информацию о текущем времени
    for i,fact in enumerate(memory_phrases):
        if len(fact) > 2 and fact[2] == 'current_time':
            del memory_phrases[i]
            break;

    #currentSecond= datetime.now().second
    current_minute = datetime.now().minute
    current_hour = datetime.now().hour
    current_time = u'Сейчас '+str(current_hour)
    if (current_hour%10) == 1:
        current_time += u' час '
    elif (current_hour%10) in [2,3,4]:
        current_time += u' часа '
    else:
        current_time += u' часов '

    current_time += str(current_minute)
    if (current_minute%10) == 1:
        current_time += u' минута '
    elif (current_minute%10) in [2,3,4]:
        current_time += u' минуты '
    else:
        current_time += u' минут '

    memory_phrases.append((current_time, '3', 'current_time'))



    question_words = tokenizer.tokenize(question)

    # Может потребоваться смена грамматического лица.
    # Сначала определим грамматическое лицо введенного предложения.
    if False:
        # Используем нейросетевой классификатор грамматического лица
        max_inputseq_len = person_change_model_config['max_inputseq_len']
        word_dims = person_change_model_config['word_dims']
        X_probe = np.zeros((1, max_inputseq_len, word_dims))
        vectorize_words( pad_wordseq(question_words, max_inputseq_len), X_probe, 0, w2v, wc2v)
        y_person = models['person_classifier'].predict(X_probe)
        person = ['1s', '2s', '3'][ np.argmax( y_person[0] ) ]
    else:
        # Для определения грамматического лица вопроса используем XGB классификатор.
        question_wx = words2str(question_words)
        shingles = ngrams(question_wx, xgb_person_classifier_shingle_len)
        X_data = lil_matrix((1, xgb_person_classifier_nb_features), dtype='bool')
        for shingle in shingles:
            X_data[0,xgb_person_classifier_shingle2id[shingle]] = True
        D_data = xgboost.DMatrix(X_data)
        y = xgb_person_classifier.predict(D_data)
        person = ['1s', '2s', '3'][ int(y[0]) ]


    if trace_enabled:
        print('detected person={}'.format(person))

    #person = get_person(question, tokenizer)
    if person=='1s':
        question = change_person(question, tokenizer, '2s')
    elif person=='2s':
        question = change_person(question, tokenizer, '1s')


    if question0[-1] == u'.':
        # Утверждение добавляем как факт в базу знаний
        fact_person = '3'
        if person == '1s':
            fact_person='2s'
        elif person == '2s':
            fact_person='1s'

        fact = question
        if trace_enabled:
            print(u'Adding [{}] to knowledge base'.format(fact))
        memory_phrases.append((fact, fact_person, '--from dialogue--'))

        continue

    if trace_enabled:
        print(u'Question to process={}'.format(question))

    question_words = tokenizer.tokenize(question)

    # определяем наиболее релевантную предпосылку
    # все предпосылки из текущей базы фактов векторизуем в один тензор, чтобы
    # прогнать его через классификатор разом.
    nb_answers = len(memory_phrases)

    if False:
        # Поиск наиболее релевантной предпосылки с помощью нейросетевой модели
        X1_probe = np.zeros((nb_answers, max_inputseq_len, word_dims), dtype=np.float32)
        X2_probe = np.zeros((nb_answers, max_inputseq_len, word_dims), dtype=np.float32)

        best_premise = ''
        best_sim = 0.0
        for ipremise, (premise, premise_person, phrase_code) in enumerate(memory_phrases):
            premise_words = tokenizer.tokenize(premise)
            vectorize_words(pad_wordseq(premise_words, max_inputseq_len), X1_probe, ipremise, w2v, wc2v)
            vectorize_words(pad_wordseq(question_words, max_inputseq_len), X2_probe, ipremise, w2v, wc2v)

        reslist = []
        y_probe = relevancy_model.predict(x={'input_words1': X1_probe, 'input_words2': X2_probe})
        for ipremise, (premise, premise_person, phrase_code) in enumerate(memory_phrases):
            sim = y_probe[ipremise]
            reslist.append( (premise, sim) )

        reslist = sorted(reslist, key=lambda z: -z[1])

        #print('\nPremise selection:')
        #for premise,sim in reslist:
        #    print(u'{}\t=>\t{}'.format(sim, premise))
        #print('\n\n')
        best_premise = reslist[0][0]
    else:
        # Поиск наиболее релевантной предпосылки с помощью XGB модели
        X_data = lil_matrix((nb_answers, xgb_relevancy_nb_features), dtype='bool')

        best_premise = ''
        best_sim = 0.0
        for ipremise, (premise, premise_person, phrase_code) in enumerate(memory_phrases):
            premise_words = tokenizer.tokenize(premise)
            question_words = tokenizer.tokenize(question)
            premise_wx = words2str(premise_words)
            question_wx = words2str(question_words)

            premise_shingles = set(ngrams(premise_wx, xgb_relevancy_shingle_len))
            question_shingles = set(ngrams(question_wx, xgb_relevancy_shingle_len))

            xgb_relevancy_vectorize_sample_x(X_data, ipremise, premise_shingles, question_shingles, xgb_relevancy_shingle2id)

        D_data = xgboost.DMatrix(X_data)
        y_probe = xgb_relevancy.predict(D_data)

        reslist = []
        for ipremise, (premise, premise_person, phrase_code) in enumerate(memory_phrases):
            sim = y_probe[ipremise]
            reslist.append( (premise, sim) )

        reslist = sorted(reslist, key=lambda z: -z[1])

        best_premise = reslist[0][0]

    if trace_enabled:
        print(u'Best premise={}'.format(best_premise))


    # Определяем способ генерации ответа
    max_wordseq_len2 = int(qa_model_config['max_inputseq_len'])
    X1_probe = np.zeros((1, max_wordseq_len2, word_dims), dtype=np.float32)
    X2_probe = np.zeros((1, max_wordseq_len2, word_dims), dtype=np.float32)
    premise_words = pad_wordseq(tokenizer.tokenize(best_premise), max_wordseq_len2)
    question_words = pad_wordseq(tokenizer.tokenize(question), max_wordseq_len2)
    vectorize_words(premise_words, X1_probe, 0, w2v, wc2v)
    vectorize_words(question_words, X2_probe, 0, w2v, wc2v)
    y_probe = models['model_selector'].predict({'input_words1': X1_probe, 'input_words2': X2_probe})
    model_selector = np.argmax( y_probe[0] )
    if trace_enabled:
        print('model_selector={}'.format(model_selector))

    answer = u''

    if model_selector == 0:
        # yes/no

        if False:
            y_probe = models['yes_no'].predict({'input_words1': X1_probe, 'input_words2': X2_probe})
            a = np.argmax(y_probe[0])
            if a == 0:
                answer = u'нет'
            else:
                answer = u'да'
        else:
            # Модель классификации ответа на базе XGB
            premise_wx = words2str(premise_words)
            question_wx = words2str(question_words)

            premise_shingles = set(ngrams(premise_wx, xgb_person_classifier_shingle_len))
            question_shingles = set(ngrams(question_wx, xgb_person_classifier_shingle_len))

            X_data = lil_matrix((1, xgb_yesno_nb_features), dtype='bool')
            xgb_yesno_vectorize_sample_x(X_data, 0, premise_shingles, question_shingles, xgb_yesno_shingle2id)

            D_data = xgboost.DMatrix(X_data, feature_names=xgb_yesno_feature_names)
            y = xgb_yesno.predict(D_data)[0]
            if y<0.5:
                answer = u'нет'
            else:
                answer = u'да'




    elif model_selector == 1:
        # word copy

        if False:
            # wordcopy #1
            # Для этой модели нужно выравнивать цепочку слов добавлением пустышек справа,
            # поэтому векторизуем заново
            premise_words = rpad_wordseq(tokenizer.tokenize(best_premise), max_wordseq_len2)
            question_words = rpad_wordseq(tokenizer.tokenize(question), max_wordseq_len2)

            X1_probe.fill(0)
            X2_probe.fill(0)

            vectorize_words(premise_words, X1_probe, 0, w2v, wc2v)
            vectorize_words(question_words, X2_probe, 0, w2v, wc2v)

            y_probe = models['word_copy'].predict({'input_words1': X1_probe, 'input_words2': X2_probe})

            #print('DEBUG X1_probe={}'.format(X1_probe[0]))
            #print('DEBUG X2_probe={}'.format(X2_probe[0]))
            #print('DEBUG y_probe={}'.format(y_probe[0]))

            words = []
            for ipremise,premise_word in enumerate(premise_words):
                if y_probe[0][ipremise]>0.5:
                    words.append(premise_word)
            answer = u' '.join(words)
        else:
            # wordcopy #3
            # эта модель имеет 3 классификатора на выходе.
            # первый классификатор выбирает позицию начала цепочки, второй - конца.
            premise_words = rpad_wordseq(tokenizer.tokenize(best_premise), max_wordseq_len2)
            question_words = rpad_wordseq(tokenizer.tokenize(question), max_wordseq_len2)

            X1_probe.fill(0)
            X2_probe.fill(0)

            vectorize_words(premise_words, X1_probe, 0, w2v, wc2v)
            vectorize_words(question_words, X2_probe, 0, w2v, wc2v)

            (y1_probe, y2_probe) = models['word_copy3'].predict({'input_words1': X1_probe, 'input_words2': X2_probe})
            beg_pos = np.argmax(y1_probe[0])
            end_pos = np.argmax(y2_probe[0])
            words = premise_words[beg_pos:end_pos+1]
            answer = u' '.join(words)

    else:
        answer = 'ERROR: answering model for {} is not implemented'.format(model_selector)

    print(u'A:> '+colorama.Fore.GREEN + u'{}'.format(answer)+colorama.Fore.RESET)


# ----------------------------------------------------------------------
#print('Bye...')
with codecs.open('../input/smalltalk_closing.txt', 'r', 'utf-8') as rdr:
    closers = []
    for line in rdr:
        if len(line)>2:
            closers.append(line.strip())

    print( np.random.choice(closers) )

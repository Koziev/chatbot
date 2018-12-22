# -*- coding: utf-8 -*-
"""
Реализация модели посимвольной генерации ответа.
"""

import os
import json
import logging
import itertools

import xgboost
from scipy.sparse import lil_matrix

from answer_generator_model import AnswerGeneratorModel


BEG_WORD = u'\b'
END_WORD = u'\n'

BEG_CHAR = u'\b'
END_CHAR = u'\n'


def ngrams(s, n):
    return [u''.join(z) for z in itertools.izip(*[s[i:] for i in range(n)])]


def words2str(words):
    """
    Цепочку слов соединяем в строку, добавляя перед цепочкой и после нее
    пробел и специальные символы начала и конца.
    :param words:
    :return:
    """
    return BEG_WORD + u' ' + u' '.join(words) + u' ' + END_WORD


class XGB_AnswerGeneratorModel(AnswerGeneratorModel):
    def __init__(self):
        super(XGB_AnswerGeneratorModel, self).__init__()
        self.logger = logging.getLogger('XGB_AnswerGeneratorModel')
        self.model = None
        self.model_config = None

    def load(self, models_folder):
        self.logger.info('Loading XGB_AnswerGeneratorModel model files')

        with open(os.path.join(models_folder, 'xgb_answer_generator.config'), 'r') as f:
            cfg = json.load(f)

        self.outshingle2id = cfg['outshingle2id']
        self.inshingle2id = cfg['inshingle2id']
        self.outchar2id = cfg['outchar2id']
        self.model_filename = cfg['model_filename']
        self.SHINGLE_LEN = cfg['shingle_len']
        self.NB_PREV_CHARS = cfg['NB_PREV_CHARS']
        self.BEG_LEN = cfg['BEG_LEN']
        self.END_LEN = cfg['END_LEN']
        self.nb_features = cfg['nb_features']

        self.phrase2sdr = None

        self.generator = xgboost.Booster()
        self.generator.load_model(self.get_model_filepath(models_folder, cfg['model_filename']))

        self.id2outchar = dict((i, c) for c, i in self.outchar2id.items())

    def vectorize_sample_x(self, X_data, idata,
                           premise_shingles, question_shingles, answer_shingles,
                           premise_beg_shingles, question_beg_shingles,
                           premise_end_shingles, question_end_shingles,
                           premise_sdr, question_sdr,
                           answer_prev_chars, word_index, char_index,
                           premise_str, premise_words,
                           question_str, question_words,
                           lexicon,
                           inshingle2id, outshingle2id, outchar2id):
        ps = set(premise_shingles)
        qs = set(question_shingles)

        common_shingles = ps & qs
        notmatched_ps = ps - qs
        notmatched_qs = qs - ps

        nb_inshingles = len(inshingle2id)

        icol = 0

        sx = [common_shingles, notmatched_ps, notmatched_qs,
              premise_beg_shingles, question_beg_shingles,
              premise_end_shingles, question_end_shingles]

        for shingles in sx:
            for shingle in shingles:
                if shingle in inshingle2id:
                    X_data[idata, icol + inshingle2id[shingle]] = True
            icol += nb_inshingles

        nb_outshingles = len(outshingle2id)
        for shingle in answer_shingles:
            if shingle in outshingle2id:
                X_data[idata, icol + outshingle2id[shingle]] = True
        icol += nb_outshingles

        for c in answer_prev_chars:
            X_data[idata, icol + outchar2id[c]] = True
        icol += self.NB_PREV_CHARS * len(outchar2id)

        X_data[idata, icol] = word_index
        icol += 1

        X_data[idata, icol] = char_index
        icol += 1

        if premise_sdr is not None:
            # for i, x in enumerate(premise_sdr):
            #    if x:
            #        X_data[idata, icol+i] = True
            X_data[idata, icol:icol + self.PHRASE_DIM] = premise_sdr[0, :]
            icol += self.PHRASE_DIM

            # for i, x in enumerate(question_sdr):
            #    if x:
            #        X_data[idata, icol+i] = True
            X_data[idata, icol:icol + self.PHRASE_DIM] = question_sdr[0, :]
            icol += self.PHRASE_DIM

        # помечаем символы, которые могут быть после последнего символа в сгенерированной
        # части ответа с точки зрения строки предпосылки, вопроса и т.д.
        prev_char1 = answer_prev_chars[::-1][-1]

        premise_str1 = premise_str.replace(BEG_CHAR + u' ', BEG_CHAR)
        for c, char_index in outchar2id.items():
            if prev_char1 + c in premise_str1:
                X_data[idata, icol + char_index] = True
        icol += len(outchar2id)

        question_str1 = question_str.replace(BEG_CHAR + u' ', BEG_CHAR)
        for c, char_index in outchar2id.items():
            if prev_char1 + c in question_str1:
                X_data[idata, icol + char_index] = True
        icol += len(outchar2id)

        premise_words_2grams = set()
        for premise_word in premise_words:
            for wordform in lexicon.get_forms(premise_word):
                premise_words_2grams.update(ngrams(u' ' + wordform + u' ', 2))
        for c, char_index in outchar2id.items():
            if prev_char1 + c in premise_words_2grams:
                X_data[idata, icol + char_index] = True
        icol += len(outchar2id)

        question_words_2grams = set()
        for question_word in question_words:
            for wordform in lexicon.get_forms(question_word):
                question_words_2grams.update(ngrams(u' ' + wordform + u' ', 2))
        for c, char_index in outchar2id.items():
            if prev_char1 + c in question_words_2grams:
                X_data[idata, icol + char_index] = True
        icol += len(outchar2id)

    def generate_answer0(self, xgb_answer_generator, tokenizer,
                         outshingle2id, inshingle2id, outchar2id,
                         shingle_len, nb_prev_chars, nb_features, id2outchar, phrase2sdr,
                         lexicon,
                         premise, question):
        premise_words = tokenizer.tokenize(premise)
        question_words = tokenizer.tokenize(question)

        premise_wx = words2str(premise_words)
        question_wx = words2str(question_words)

        premise_shingles = ngrams(premise_wx, shingle_len)
        question_shingles = ngrams(question_wx, shingle_len)

        premise_beg_shingles = ngrams(premise_wx[:self.BEG_LEN], self.SHINGLE_LEN)
        question_beg_shingles = ngrams(question_wx[:self.BEG_LEN], self.SHINGLE_LEN)

        premise_end_shingles = ngrams(premise_wx[-self.END_LEN:], self.SHINGLE_LEN)
        question_end_shingles = ngrams(question_wx[-self.END_LEN:], self.SHINGLE_LEN)

        if phrase2sdr is not None:
            premise_sdr = phrase2sdr[premise_wx]
            question_sdr = phrase2sdr[question_wx]
        else:
            premise_sdr = None
            question_sdr = None

        answer_chain = BEG_CHAR

        while True:
            # цикл добавления новых сгенерированных символов
            answer_len = len(answer_chain)
            answer_shingles = ngrams(answer_chain, shingle_len)
            answer_prev_chars = answer_chain[max(0, answer_len - nb_prev_chars):answer_len]
            answer_prev_chars = answer_prev_chars[::-1]

            left_chars = answer_chain[1:]

            # номер генерируемого слова получаем как число пробелов слева
            word_index = left_chars.count(u' ')

            # номер генерируемого символа в генерируемом слове - отсчитываем от последнего пробела
            rpos = left_chars.rfind(u' ')
            if rpos == -1:
                # это первое слово
                char_index = len(left_chars)
            else:
                char_index = len(left_chars) - rpos - 1

            X_data = lil_matrix((1, nb_features), dtype='float')
            self.vectorize_sample_x(X_data, 0, premise_shingles, question_shingles, answer_shingles,
                                    premise_beg_shingles, question_beg_shingles,
                                    premise_end_shingles, question_end_shingles,
                                    premise_sdr, question_sdr,
                                    answer_prev_chars, word_index, char_index,
                                    premise_wx, premise_words,
                                    question_wx, question_words,
                                    lexicon,
                                    inshingle2id, outshingle2id, outchar2id)

            D_data = xgboost.DMatrix(X_data, silent=True)
            y = xgb_answer_generator.predict(D_data)
            c = id2outchar[y[0]]
            answer_chain += c
            if c == END_CHAR or answer_len >= 100:
                break

        return u'{}'.format(answer_chain[1:-1]).strip()

    def generate_answer(self, premise_str, question_str, text_utils, word_embeddings):
        answer = self.generate_answer0(self.generator, text_utils,
                                       self.outshingle2id, self.inshingle2id, self.outchar2id,
                                       self.SHINGLE_LEN, self.NB_PREV_CHARS, self.nb_features, self.id2outchar, self.phrase2sdr,
                                       text_utils.get_lexicon(),
                                       premise_str, question_str)

        return answer

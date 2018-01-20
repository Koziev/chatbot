# -*- coding: utf-8 -*-

from base_answering_machine import BaseAnsweringMachine
from simple_dialog_session_factory import SimpleDialogSessionFactory

from keras.layers import Embedding
from keras.models import model_from_json
from keras.layers.merge import concatenate, add, multiply
from keras.layers import Lambda
from keras import backend as K
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.layers.core import Activation, RepeatVector, Dense, Masking
from keras.layers.wrappers import Bidirectional
from keras.layers import Input
import keras.callbacks
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
import xgboost
from scipy.sparse import lil_matrix
import json
import os
import pickle
import gensim
import numpy as np
import logging


class WordEmbeddings(object):
    """
    Загрузка и работа с векторными моделями слов.
    """

    def __init__(self):
        pass

    def load_models(self, w2v_path, wc2v_path):
        logging.info(u'Loading wordchar2vector from {}'.format(wc2v_path))
        self.wc2v = gensim.models.KeyedVectors.load_word2vec_format(wc2v_path, binary=False)
        self.wc2v_dims = len(self.wc2v.syn0[0])

        logging.info(u'Loading word2vector from {}'.format(w2v_path))
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
        self.w2v_dims = len(self.w2v.syn0[0])

    def get_dims(self):
        return self.wc2v_dims+self.w2v_dims

    def vectorize_words(self, words, X_batch, irow):
        for iword, word in enumerate(words):
            if word in self.w2v:
                X_batch[irow, iword, :self.w2v_dims] = self.w2v[word]
            if word in self.wc2v:
                X_batch[irow, iword, self.w2v_dims:] = self.wc2v[word]

# -*- coding: utf-8 -*-

import os
import gensim
import logging

from wordchar2vector_model import Wordchar2VectorModel
from text_utils import PAD_WORD


class WordEmbeddings(object):
    """
    Загрузка и работа с векторными моделями слов.
    """

    def __init__(self):
        self.wc2v = None
        self.wc2v_dims = None
        self.w2v = dict()
        self.w2v_dims = dict()
        self.wordchar2vector_model = None

    def load_models(self, models_folder):
        """
        Загружаются нейросетевые модели, позволяющие сгенерировать
        вектор нового слова. Для самых частотных слов готовые вектора
        рассчитаны заранее и сохранены в файле, поэтому они будут
        обработаны объектом self.wc2v.
        """
        self.wordchar2vector_model = Wordchar2VectorModel()
        self.wordchar2vector_model.load(models_folder)

    def load_wc2v_model(self, wc2v_path):
        logging.info(u'Loading wordchar2vector from {}'.format(wc2v_path))
        self.wc2v = gensim.models.KeyedVectors.load_word2vec_format(wc2v_path, binary=False)
        self.wc2v_dims = len(self.wc2v.syn0[0])

    def load_w2v_model(self, w2v_path):
        w2v_filename = os.path.basename(w2v_path)
        if w2v_filename not in self.w2v:
            logging.info(u'Loading word2vector from {}'.format(w2v_path))
            w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=not w2v_path.endswith('.txt'))
            # При обучении и при предсказании пути к w2v данным могут отличаться, так как
            # тренеры и сами боты работают на разных машинах. Поэтому селектируем по имени файла, без пути.
            self.w2v[w2v_filename] = w2v
            self.w2v_dims[w2v_filename] = len(w2v.syn0[0])

    def vectorize_words(self, w2v_filename, words, X_batch, irow):
        w2v = self.w2v[w2v_filename]
        w2v_dims = self.w2v_dims[w2v_filename]
        for iword, word in enumerate(words):
            if word != PAD_WORD:
                if word in w2v:
                    X_batch[irow, iword, :w2v_dims] = w2v[word]
                if word in self.wc2v:
                    X_batch[irow, iword, w2v_dims:] = self.wc2v[word]
                else:
                    X_batch[irow, iword, w2v_dims:] = self.wordchar2vector_model.build_vector(word)

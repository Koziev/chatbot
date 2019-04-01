# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import gensim
import os
import sys
import logging
from gensim.models.wrappers import FastText


class WordEmbeddings(object):
    def __init__(self):
        self.w2v_dims = -1
        self.wc2v_dims = -1
        self.vector_size = -1
        self.w2v = None
        self.wc2v = None

    @staticmethod
    def _is_int(word):
        """Вернет True, если токен 'word' является числом и не может быть найден в w2v модели"""
        return word.isdigit()

    def __getitem__(self, word):
        raise NotImplementedError()

    @staticmethod
    def _flush_print():
        """Сброс буфера вывода консоли - чтобы уведомление о загрузки модели без \r появилось сразу."""
        sys.stdout.flush()

    @staticmethod
    def load_word_vectors(wordchar2vector_path, word2vector_path):
        """
        Фабрика для получения удобного доступа к обеим моделям встраивания слов - посимвольной
        морфологической и пословной синтактико-семантической.

         :param wordchar2vector путь к файлу с векторами слов в модели посимвольного встраивания.
         :param word2vector_path путь к файлу с векторами слов в word2vec, fasttext или glove моделях

         :return экземпляр класса, предоставляющий метод-индексатор и возвращающий объединенный
          вектор встраивания для слова.
        """

        logging.info('Loading the wordchar2vector model {} '.format(wordchar2vector_path), end='')
        # Грузим заранее подготовленные векторы слов для модели
        # встраивания wordchar2vector (см. wordchar2vector.py)
        wc2v = gensim.models.KeyedVectors.load_word2vec_format(wordchar2vector_path, binary=False)
        wc2v_dims = len(wc2v.syn0[0])
        logging.debug('wc2v_dims={0}'.format(wc2v_dims))

        if os.path.basename(word2vector_path).startswith('fasttext'):
            logging.info('Loading FastText model {} '.format(word2vector_path), end='')
            WordEmbeddings._flush_print()
            w2v = FastText.load_fasttext_format(word2vector_path)
            w2v_dims = w2v.vector_size
            print('w2v_dims={0}'.format(w2v_dims))
            return WordEmbeddings_FastText(wc2v, wc2v_dims, w2v, w2v_dims)
        else:
            logging.info('Loading w2v model {} '.format(word2vector_path), end='')
            WordEmbeddings._flush_print()
            w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vector_path, binary=not word2vector_path.endswith('.txt'))
            w2v_dims = len(w2v.syn0[0])
            logging.debug('w2v_dims={0}'.format(w2v_dims))
            return WordEmbeddings_W2V(wc2v, wc2v_dims, w2v, w2v_dims)

    def all_words_known(self, words):
        return all((word in self.wc2v) for word in words)




class WordEmbeddings_W2V(object):
    """
    Класс предоставляет удобный индексатор для модели встраивания word2vector,
    обрабатывая ситуации отсутствия слова и числа. Экземпляр этого класса
    создается и возвращается методом WordEmbeddings.load_word_vectors(...).
    """
    def __init__(self, wc2v, wc2v_dims, w2v, w2v_dims):
        self.w2v_dims = w2v_dims
        self.wc2v_dims = wc2v_dims
        self.vector_size = w2v_dims + wc2v_dims
        self.w2v = w2v
        self.wc2v = wc2v
        self.missing_wc2c = set()

    def __contains__(self, word):
        return word != u''

    def __getitem__(self, word):
        v = np.zeros(self.vector_size)
        if WordEmbeddings._is_int(word) and u'_num_' in self.w2v:
            v[:self.w2v_dims] = self.w2v[u'_num_']
        elif word in self.w2v:
            v[:self.w2v_dims] = self.w2v[word]
        else:
            pass

        if word not in self.wc2v:
            if word not in self.missing_wc2c:
                logging.error(u'Word {} missing in wordchar2vector model'.format(word))
                self.missing_wc2c.add(word)
        else:
            v[self.w2v_dims:] = self.wc2v[word]
        return v

    def all_words_known(self, words):
        return not any((w not in self.wc2v) for w in words)


class WordEmbeddings_FastText(object):
    """
    Класс предоставляет удобный индексатор для модели встраивания FastText,
    корректно обрабатывая ситуации отсутствия слова. Экземпляр этого класса
    создается и возвращается методом WordEmbeddings.load_word_vectors(...).
    """
    def __init__(self, wc2v, wc2v_dims, w2v, w2v_dims):
        self.w2v_dims = w2v_dims
        self.wc2v_dims = wc2v_dims
        self.vector_size = w2v_dims + wc2v_dims
        self.w2v = w2v
        self.wc2v = wc2v

    def __getitem__(self, word):
        v = np.zeros(self.vector_size)
        if WordEmbeddings._is_int(word):
            v[:self.w2v_dims] = self.w2v[u'_num_']
        else:
            try:
                v[:self.w2v_dims] = self.w2v[word]
            except KeyError:
                pass

        v[self.w2v_dims:] = self.wc2v[word]
        return v

# -*- coding: utf-8 -*-
"""
Нейросетевая реализация модели генерации ответа через выбор
копируемых слов их предпосылки. 
"""

import os
import json
import numpy as np
import logging
from keras.models import model_from_json
from word_copy_model import WordCopyModel

class NN_WordCopy3(WordCopyModel):
    def __init__(self):
        super(NN_WordCopy3, self).__init__()
        self.logger = logging.getLogger('NN_WordCopy3')
        self.model = None
        self.model_config = None

    def load(self, models_folder):
        self.logger.info('Loading NN_WordCopy3 model files')

        arch_filepath = os.path.join(models_folder, 'nn_wordcopy3.arch')
        weights_path = os.path.join(models_folder, 'nn_wordcopy3.weights')
        with open(arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(weights_path)
        self.model = m

        with open(os.path.join(models_folder, 'nn_wordcopy3.config'), 'r') as f:
            self.model_config = json.load(f)

        self.word_dims = self.model_config['word_dims']
        self.w2v_path = self.model_config['w2v_path']
        self.max_wordseq_len = int(self.model_config['max_inputseq_len'])
        self.padding = self.model_config['padding']
        self.X1_probe = np.zeros((1, self.max_wordseq_len, self.word_dims), dtype=np.float32)
        self.X2_probe = np.zeros((1, self.max_wordseq_len, self.word_dims), dtype=np.float32)
        self.w2v_filename = os.path.basename(self.w2v_path)

    def generate_answer(self, premise_str, question_str, text_utils, word_embeddings):
        # определяем диапазон слов начала и конца копируемой цепочки слов
        self.X1_probe.fill(0)
        self.X2_probe.fill(0)

        if self.padding == 'right':
            premise_words = text_utils.rpad_wordseq(text_utils.tokenize(premise_str), self.max_wordseq_len)
            question_words = text_utils.rpad_wordseq(text_utils.tokenize(question_str), self.max_wordseq_len)
        else:
            premise_words = text_utils.lpad_wordseq(text_utils.tokenize(premise_str), self.max_wordseq_len)
            question_words = text_utils.lpad_wordseq(text_utils.tokenize(question_str), self.max_wordseq_len)

        word_embeddings.vectorize_words(self.w2v_filename, premise_words, self.X1_probe, 0)
        word_embeddings.vectorize_words(self.w2v_filename, question_words, self.X2_probe, 0)

        #for i1, word1 in enumerate(premise_words):
        #    if len(word1)>0:
        #        print(u'{} {} ==> {}'.format(i1, word1, self.X1_probe[0, i1, :]))

        (y1_probe, y2_probe) = self.model.predict({'input_words1': self.X1_probe, 'input_words2': self.X2_probe})
        beg_pos = np.argmax(y1_probe[0])
        end_pos = np.argmax(y2_probe[0])
        words = premise_words[beg_pos:end_pos + 1]
        answer = u' '.join(words)
        #print(u'\nDEBUG nn_wordcopy3 @63 ==> beg_pos={} end_pos={} answer="{}"'.format(beg_pos, end_pos, answer))

        return answer

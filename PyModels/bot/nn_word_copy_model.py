# -*- coding: utf-8 -*-

import os
import logging
import json
import numpy as np
from keras.models import model_from_json
from word_copy_model import WordCopyModel

class NN_WordCopyModel(WordCopyModel):
    """
    Аппликатор нейросетевой модели генерации ответа на вопрос через
    копирование слов из предпосылки.
    """
    def __init__(self):
        super(NN_WordCopyModel, self).__init__()
        self.logger = logging.getLogger('NN_WordCopyModel')

    def load(self, models_folder):
        self.logger.info('Loading NN_WordCopyModel model files')

        arch_filepath = os.path.join(models_folder, 'qa_word_copy3.arch')
        weights_path = os.path.join(models_folder, 'qa_word_copy3.weights')
        with open(arch_filepath, 'r') as f:
            m = model_from_json(f.read())

        m.load_weights(weights_path)
        self.model = m

        with open(os.path.join(models_folder, 'qa_model.config'), 'r') as f:
            self.model_config = json.load(f)

        self.word_dims = self.model_config['word_dims']
        self.max_wordseq_len = int(self.model_config['max_inputseq_len'])
        self.X1_probe = np.zeros((1, self.max_wordseq_len, self.word_dims), dtype=np.float32)
        self.X2_probe = np.zeros((1, self.max_wordseq_len, self.word_dims), dtype=np.float32)

    def copy_words(self, premise_words, question_words, text_utils, word_embeddings):
        # эта модель имеет 2 классификатора на выходе.
        # первый классификатор выбирает позицию начала цепочки, второй - конца.
        # предполагается, что копируется непрерывная цепочка слов.

        self.X1_probe.fill(0)
        self.X2_probe.fill(0)
        padded_premise_words = text_utils.rpad_wordseq(premise_words, self.max_wordseq_len)
        padded_question_words = text_utils.rpad_wordseq(question_words, self.max_wordseq_len)

        self.vectorize_words(padded_premise_words, self.X1_probe, 0)
        self.vectorize_words(padded_question_words, self.X2_probe, 0)

        (y1_probe, y2_probe) = self.model.predict({'input_words1': self.X1_probe, 'input_words2': self.X2_probe})
        beg_pos = np.argmax(y1_probe[0])
        end_pos = np.argmax(y2_probe[0])
        words = premise_words[beg_pos:end_pos + 1]
        answer = u' '.join(words)
        return answer



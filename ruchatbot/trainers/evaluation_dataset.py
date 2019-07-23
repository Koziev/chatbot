# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import codecs
import os
import numpy as np

from evaluation_group import EvaluationGroup


class EvaluationDataset(object):
    def __init__(self, max_wordseq_len, tokenizer, padding):
        self.max_wordseq_len = max_wordseq_len
        self.tokenizer = tokenizer
        self.eval_data = []  # список из EvaluationGroup
        self.all_premises = []
        self.padding = padding

    def load(self, data_folder):
        eval_path = os.path.join(data_folder, 'evaluate_relevancy.txt')
        self.eval_data = []  # список из EvaluationGroup
        with codecs.open(eval_path, 'r', 'utf-8') as rdr:
            while True:
                group = EvaluationGroup(self.max_wordseq_len, self.tokenizer, self.padding)
                eof_reached = group.load(rdr)
                if eof_reached:
                    break
                if not group.is_empty():
                    self.eval_data.append(group)

        # Нам нужен набор нерелевантных предпосылок.
        self.all_premises = []

        # Берем нерелевантные предпосылки из базы фактов чат-бота
        with codecs.open(os.path.join(data_folder, 'premises.txt'), 'r', 'utf-8') as rdr:
            for line in rdr:
                line = line.strip()
                if len(line) > 0:
                    premise = u' '.join(self.tokenizer.tokenize(line)).strip()
                    self.all_premises.append(premise)

    def get_all_phrases(self):
        phrases = set()

        for irecord, record in enumerate(self.eval_data):
            phrases.update(u' '.join(words) for words in record.premises)
            phrases.update(u' '.join(words) for words in record.questions)

        for words in self.all_premises:
            phrases.add(u' '.join(words))

        return phrases

    def generate_groups(self):
        for irecord, record in enumerate(self.eval_data):
            for right_premise in record.premises_str:
                for question in record.questions:
                    # Отдельно обрабатываем каждую релевантную пару предпосылка-вопрос.
                    # Добавляем к ней множество предпосылок из тренировочного датасета, считая их нерелевантными.
                    phrases = []

                    # ожидаемый выбор добавляем.
                    phrases.append((right_premise, question))

                    # добавляем негативные предпосылки, которые не должны быть выбраны
                    for neg_premise in self.all_premises:
                        if neg_premise == right_premise:
                            # тот случай, когда в тренировочном датасете встретилась релевантная предпосылка.
                            pass
                        else:
                            phrases.append((neg_premise, question))

                    # перемешаем фразы, чтобы верная предпосылка находилась на разных позициях
                    phrases = list(np.random.permutation(phrases))
                    yield (irecord, phrases)

    def is_relevant_premise(self, irecord, selected_premise):
        """
        Вернет True, если предпосылка selected_premise корректна для группы irecord.
        """
        return self.eval_data[irecord].is_relevant_premise(selected_premise)

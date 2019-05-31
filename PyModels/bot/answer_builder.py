# -*- coding: utf-8 -*-
"""
Группа моделей, выполняющих генерацию текста ответа при заданных текстах предпосылки
и вопроса.

Для проекта чат-бота https://github.com/Koziev/chatbot

15-05-2019 Добавлена генеративная модель построения ответа ("Вероятностная Машина Хомского")
"""

import logging
import itertools
import os
import io

#from nn_yes_no_model import NN_YesNoModel
from bot.xgb_yes_no_model import XGB_YesNoModel
from bot.nn_model_selector import NN_ModelSelector
from bot.nn_wordcopy3 import NN_WordCopy3
from bot.xgb_answer_generator_model import XGB_AnswerGeneratorModel

from generative_grammar.generative_grammar_engine import GenerativeGrammarEngine
from generative_grammar.word_selector import WordSelector
from generative_grammar.answer_length_predictor import AnswerLengthPredictor
from generative_grammar.answer_relevancy import AnswerRelevancy


class AnswerBuilder(object):
    def __init__(self):
        self.logger = logging.getLogger('AnswerBuilder')
        self.trace_enabled = True
        self.grammar = None  # :GenerativeGrammarEngine генеративная грамматика для построения ответа
        self.word_selector = None  # :WordSelector модель внимания для слов
        self.len_predictor = None  # :AnswerLengthPredictor длина ответа
        self.answer_relevancy = None  # :AnswerRelevancy  релевантность сгенерированного ответа
        self.known_words = None  # set

    def load_models(self, models_folder):
        self.models_folder = models_folder

        # Модель для выбора ответов yes|no на базе XGB
        self.yes_no_model = XGB_YesNoModel()
        self.yes_no_model.load(models_folder)

        #self.yes_no_model = NN_YesNoModel()
        #self.yes_no_model.load(models_folder)

        # Модель для выбора способа генерации ответа
        self.model_selector = NN_ModelSelector()
        self.model_selector.load(models_folder)

        # нейросетевые модели для генерации ответа.
        self.word_copy_model = NN_WordCopy3()
        self.word_copy_model.load(models_folder)

        self.answer_generator = XGB_AnswerGeneratorModel()
        self.answer_generator.load(models_folder)

        self.grammar = GenerativeGrammarEngine()
        self.grammar.load(models_folder)

        self.word_selector = WordSelector()
        self.word_selector.load(models_folder)

        self.len_predictor = AnswerLengthPredictor()
        self.len_predictor.load(models_folder)

        self.answer_relevancy = AnswerRelevancy()
        self.answer_relevancy.load(models_folder)

        self.load_known_words(os.path.join(models_folder, 'dataset_words.txt'))

    def load_known_words(self, file_path):
        # По этому списку слов будет отсекать всякую экзотичную лексику
        self.known_words = set()
        with io.open(file_path, 'r', encoding='utf-8') as rdr:
            for line in rdr:
                word = line.strip()
                self.known_words.add(word)

    def get_w2v_paths(self):
        paths = set()

        if self.word_copy_model.w2v_path is not None:
            #logging.info('word_copy_model requires {}'.format(self.word_copy_model.get_w2v_path()))
            paths.add(self.word_copy_model.get_w2v_path())

        if self.model_selector.w2v_path is not None:
            #logging.info('model_selector requires {}'.format(self.model_selector.get_w2v_path()))
            paths.add(self.model_selector.get_w2v_path())

        if self.yes_no_model.w2v_path is not None:
            #logging.info('yes_no_model requires {}'.format(self.yes_no_model.get_w2v_path()))
            paths.add(self.yes_no_model.get_w2v_path())

        if self.answer_generator.w2v_path is not None:
            #logging.debug('answer_generator requires {}'.format(self.answer_generator.get_w2v_path()))
            paths.add(self.answer_generator.get_w2v_path())

        if self.word_selector is not None:
            paths.add(self.word_selector.get_w2v_path())

        return list(paths)

    def build_answer_text(self, premise_groups, premise_rels, question, text_utils, word_embeddings):
        # Определяем способ генерации ответа
        answers = []
        answer_rels = []

        for premises, premise_rel in itertools.izip(premise_groups, premise_rels):
            assert(len(premises) <= 1)
            premise = premises[0] if len(premises) == 1 else u''

            model_selector = self.model_selector.select_model(premise_str_list=premises,
                                                              question_str=question,
                                                              text_utils=text_utils,
                                                              word_embeddings=word_embeddings)
            if self.trace_enabled:
                self.logger.debug('model_selector={}'.format(model_selector))

            # Теперь применяем соответствующую модель к предпосылкам и вопросу.
            answer = u''

            if model_selector == 0:
                # Ответ генерируется через классификацию на 2 варианта yes|no
                y = self.yes_no_model.calc_yes_no(premises, question, text_utils, word_embeddings)
                if y < 0.5:
                    answer = text_utils.language_resources[u'not']
                else:
                    answer = text_utils.language_resources[u'yes']
                answers.append(answer)
                answer_rels.append(premise_rel)
                break  # ответ да/нет всегда единственный
            else:
                bad_tokens = set(u'? . !'.split())

                tokenized_premises = [list(filter(lambda w: w not in bad_tokens, text_utils.tokenize(prem.lower())))
                                      for prem
                                      in premises]
                tokenized_question = list(filter(lambda w: w not in bad_tokens, text_utils.tokenize(question.lower())))

                # Внимание на слова
                word_p = self.word_selector.select_words(tokenized_premises, tokenized_question, word_embeddings)

                if self.trace_enabled:
                    self.logger.debug('Selected words and their weights:')
                    for word, p in sorted(word_p, key=lambda z: -z[1]):
                        self.logger.debug(u'{:15s}\t{}'.format(word, p))

                # Определяем распределение вероятности для вариантов длин ответа
                len2proba = self.len_predictor.predict(tokenized_premises, tokenized_question, word_embeddings)
                # начало отладки
                max_len_p = 0.0
                best_len = 0
                for l, p in len2proba.items():
                    if p > max_len_p:
                        max_len_p = p
                        best_len = l

                if self.trace_enabled:
                    self.logger.debug(u'Most probable answer length={} (p={})'.format(best_len, max_len_p))
                    # print('Answer length probabilities:')
                    # for l, p in len_p.items():
                    #    print('len={} p={}'.format(l, p))
                    # конец отладки

                answer = None
                answer_rel = 0.0

                if model_selector == 3:
                    # Вариант 3 - особый случай, когда выдается строка из одних цифр
                    # Тут можно использовать или посимвольную генерацию общего назначения,
                    # или специализированную модель.
                    # Сейчас используется общая модель.
                    answer = self.answer_generator.generate_answer(premise,
                                                                   question,
                                                                   text_utils,
                                                                   word_embeddings)
                    answer_rel = premise_rel
                elif model_selector == 1:
                    # ответ создается через копирование слов из предпосылки.
                    answer1 = self.word_copy_model.generate_answer(premise,
                                                                   question,
                                                                   text_utils,
                                                                   word_embeddings)
                    answer1_rel = 0.0
                    if len(answer1) > 0:
                        answer1_rel = self.answer_relevancy.score_answer(tokenized_premises, tokenized_question,
                                                                         answer1.split(), word_embeddings)
                    else:
                        self.logger.error(
                            u'Empty answer generated by word_copy_model for premise={}, question={}'.format(premise,
                                                                                                            question))
                    answer = answer1
                    answer_rel = answer1_rel

                    # Альтернативный подход генерации через копирование
                    if answer1_rel < 0.5:
                        if len(premises) == 1:
                            best_words = sorted(word_p, key=lambda wp: -wp[1])[:best_len]
                            if all((word in tokenized_premises[0]) for word, p in best_words):  # все выбранные слова находятся в предпосылке
                                # сортируем выбранные слова так, как они идут в предпосылке
                                word2index = dict(enumerate(tokenized_premises[0]))
                                sorted_best_words = [word for word, p in sorted(best_words, key=lambda wp: word2index.get(wp[0], -1))]
                                answer2 = text_utils.build_output_phrase(sorted_best_words)
                                answer2_rel = self.answer_relevancy.score_answer(tokenized_premises, tokenized_question,
                                                                                 sorted_best_words, word_embeddings)

                                if answer2_rel > answer1_rel:
                                    # Второй вариант ответа имеет лучшее качество, берем его.
                                    answer = answer2
                                    answer_rel = answer2_rel

                if answer_rel < 0.5 or model_selector == 2:
                    # Ответ генерируется либо машиной Хомского, либо посимвольно.

                    # Запускаем генерацию через МХ
                    if self.trace_enabled:
                        self.logger.debug('Start generating answers via Chomsky machine')

                    # Оптимизация от 20-05-2019:
                    # 1) отбрасываем слишком малозначимые слова
                    p_threshold = max(p for word, p in word_p) * 0.02
                    word_p = [(word, p) for word, p in word_p if p > p_threshold]
                    # 2) если слов все равно осталось много, то оставим максимальную длину + 1
                    if len(word_p) > (best_len + 1):
                        word_p = sorted(word_p, key=lambda z: -z[1])[:best_len + 1]

                    all_generated_phrases = self.grammar.generate2(word_p, self.known_words)

                    if self.trace_enabled:
                        self.logger.debug('Ranking {} answers'.format(len(all_generated_phrases)))

                    if len(all_generated_phrases) > 0:
                        scored_answers = self.answer_relevancy.score_answers(tokenized_premises, tokenized_question,
                                                                      all_generated_phrases, word_embeddings,
                                                                      text_utils, len2proba)

                        sorted_answers = sorted(scored_answers, key=lambda z: -z.get_rank())

                        if self.trace_enabled:
                            self.logger.debug(u'Best generated answer is {} p={}'.format(sorted_answers[0].get_str(),
                                                                                         sorted_answers[0].get_rank()))
                        #    for phrase in sorted_answers[:10]:
                        #        print(u'{:6f}\t{}'.format(phrase.get_rank(), phrase.get_str()))

                        answer3 = sorted_answers[0].get_str()
                        answer3_rel = sorted_answers[0].get_rank()
                    else:
                        answer3 = None
                        answer3_rel = 0.0

                    # Теперь посимвольная генерация
                    answer4 = self.answer_generator.generate_answer(premise,
                                                                    question,
                                                                    text_utils,
                                                                    word_embeddings)
                    answer4_rel = self.answer_relevancy.score_answer(tokenized_premises, tokenized_question,
                                                                     answer4.split(), word_embeddings)
                    if answer4_rel > answer3_rel:
                        answer = answer4
                        answer_rel = answer4_rel
                    else:
                        answer = answer3
                        answer_rel = answer3_rel


                if len(answer) > 0:
                    if answer not in answers:
                        answers.append(answer)
                        answer_rels.append(premise_rel*answer_rel)

                break

        return answers, answer_rels

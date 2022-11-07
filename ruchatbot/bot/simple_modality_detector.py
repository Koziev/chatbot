# -*- coding: utf-8 -*-

from ruchatbot.bot.modality_detector import ModalityDetector


class SimpleModalityDetectorRU(ModalityDetector):
    """
    Временная затычка для модели определения модальности.
    Работает через проверку наличия в начале фразы ключевых слов типа ПОЧЕМУ.
    Детектор классифицирует фразы на утверждения, вопросы и императивы.
    """
    def __init__(self):
        pass

    def get_modality(self, phrase, text_utils):
        """
        Определяем параметры модальности для фразы.
        Вернем пару значений: тип фразы (утверждение, вопрос, приказ) + лицо (не всегда, только для утверждений)
        """
        #word_embeddings = text_utils.word_embeddings

        person = -1

        words = text_utils.tokenize(phrase)

        if person == -1:
            person = text_utils.detect_person0(words)

        if len(phrase) == 0:
            return ModalityDetector.undefined, person, words
        elif phrase[-1] == u'?':
            return ModalityDetector.question, person, words

        # Фразы, оканчивающиеся на "!", могут быть как императивами, так и эмоционально
        # выраженными утверждениями:
        # 1) Закажи пиццу!
        # 2) Ты молодец!
        # Поэтому нам нужны результаты частеречной разметки, чтобы разделить эти случаи.

        # Проверяем наличие вопросительных слов, которые могут быть в любом месте фразы: "а ты кто"
        if any(text_utils.is_question_word(word) for word in words):
            return ModalityDetector.question, person, words

        if len(words) > 1 and text_utils.is_question_word(words[1]):
            return ModalityDetector.question, person, words

        # Определение императивных форм глаголов требует проведения частеречной
        # разметки, чтобы снять неоднозначности типа МОЙ/МЫТЬ
        tags = list(text_utils.tag(words))

        if phrase[-1] == '!':
            # Если есть глагол, то считаем императивом
            if any((u'VERB' in tag) for (word, tag) in tags):
                return ModalityDetector.imperative, person, words

        if any((u'Mood=Imp' in tag) for (word, tag) in tags):
            return ModalityDetector.imperative, person, words

        if any((u'Person=2' in tag) for (word, tag) in tags):
            person = 2

        return ModalityDetector.assertion, person, words

    def load(self, models_folder):
        # nothing to do
        pass

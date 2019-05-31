# -*- coding: utf-8 -*-

from bot.modality_detector import ModalityDetector


class SimpleModalityDetectorRU(ModalityDetector):
    """
    Временная затычка для модели определения модальности.
    Работает через проверку наличия в начале фразы ключевых слов типа ПОЧЕМУ.
    """
    def __init__(self):
        pass

    def get_modality(self, phrase, text_utils, word_embeddings):
        if len(phrase) == 0:
            return ModalityDetector.undefined
        elif phrase[-1] == u'?':
            return ModalityDetector.question
        elif phrase[-1] == u'!':
            return ModalityDetector.imperative

        words = text_utils.tokenize(phrase)
        if text_utils.is_question_word(words[0]):
            return ModalityDetector.question

        if len(words) > 1 and text_utils.is_question_word(words[1]):
            return ModalityDetector.question

        # Определение императивных форм глаголов требует проведения частеречной
        # разметки, чтобы снять неоднозначности типа МОЙ/МЫТЬ
        tags = text_utils.tag(words)
        if any((u'Mood=Imp' in tag) for (word, tag) in tags):
            return ModalityDetector.imperative

        return ModalityDetector.assertion

    def load(self, models_folder):
        # nothing to do
        pass

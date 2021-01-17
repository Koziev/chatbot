"""
Реализация модели для выделения признаков гендерной самоидентификации из сообщения собеседника.
TODO: подумать о переходе на полностью нейросетевую модель, чтобы учесть больше разных конструкций.
"""

import pickle
import io
import os


class InterlocutorGenderDetector:
    def __init__(self):
        self.name2gender = None

    def load(self, models_dir):
        with open(os.path.join(models_dir, 'names.pkl'), 'rb') as f:
            self.name2gender = pickle.load(f)

    def detect_interlocutor_gender(self, text_str, text_utils):
        # Пол собеседника пока неизвестен, будем пытаться определить его из лексического и синтаксического
        # содержания фразы.
        parsed_data = text_utils.parse_syntax(text_str)

        # Русскоязычные диалоги допускают несколько способов передать гендерную самоидентификацию.
        # Проверим самые частотные.
        interlocutor_gender = None

        # 1. Если есть глагол в прошедшем времени в роли сказуемого и подлежащее "я", то берем его тэг грамматического
        # рода.
        up_words = [z.form.lower().replace('ё', 'е') for z in parsed_data]
        up_lemmas = [z.lemma.lower().replace('ё', 'е') for z in parsed_data]
        edges2 = []
        for pred_token in parsed_data:
            if pred_token.head != '0':
                edges2.append((pred_token.form.lower(), parsed_data[pred_token.head].form.lower()))

            if pred_token.head == '0' and pred_token.upos == 'VERB':
                if text_utils.get_udpipe_attr(pred_token, 'Tense') == 'Past':
                    interlocutor_gender = text_utils.get_udpipe_attr(pred_token, 'Gender')
                    if interlocutor_gender:
                        return interlocutor_gender

        # Конструкция "я должен/должна ... "
        if 'ты' in up_words and 'должен' in up_lemmas:
            if ('ты', 'должен') in edges2:
                return 'Masc'
            elif ('ты', 'должна') in edges2:
                return 'Fem'

        # Реплики с шаблоном "меня зовут Марина"
        if 'тебя' in up_words and 'звать' in up_lemmas:
            for word in up_words:
                uword = word.lower().replace('ё', 'е')
                if uword in self.name2gender:
                    g = self.name2gender[uword]
                    if g == 'm':
                        return 'Masc'
                    elif g == 'f':
                        return 'Fem'

                    return None

        # Реплики с шаблоном "мое имя Олег"
        if 'твое' in up_words and 'имя' in up_lemmas and ('твое', 'имя') in edges2:
            for word in up_words:
                uword = word.lower().replace('ё', 'е')
                if uword in self.name2gender:
                    g = self.name2gender
                    if g == 'm':
                        return 'Masc'
                    elif g == 'f':
                        return 'Fem'

                    return None

        return None

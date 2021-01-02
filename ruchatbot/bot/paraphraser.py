"""
Модель генеративной перефразировки реплик бота.
"""

import collections
import random
import os
import logging
import pickle


class Paraphraser:
    def __init__(self):
        self.processed_phrases = collections.Counter()
        self.paraphraser_templates = None
        self.simple_paraphrases = None

    def load(self, models_folder):
        with open(os.path.join(models_folder, 'generative_paraphraser.dat'), 'rb') as f:
            self.paraphraser_templates = pickle.load(f)
            self.simple_paraphrases = pickle.load(f)

    def reset_usage_stat(self):  # TODO - надо бы хранить статистику в разрезе user_id
        self.processed_phrases.clear()

    def match_support_template(self, template, tokens, w2v):
        if len(template) == len(tokens):
            match1 = dict()
            for template_item, token in zip(template, tokens):
                if template_item[1] is not None:
                    if not all((tag in token[1]) for tag in template_item[1]):
                        return None

                loc = template_item[2]
                if template_item[0] == token[0]:
                    # формы слов совпали буквально
                    if loc is not None:
                        match1[loc] = token[2]
                else:
                    sim = w2v.word_similarity(template_item[0], token[0])
                    if sim >= 0.90:
                        # близкие векторы слов в шаблоне и фразе
                        if loc is not None:
                            match1[loc] = token[2]
                    else:
                        return None

            return match1

        return None

    def is_important_token2(self, t):
        pos = t[1].split('|')[0]
        if pos in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'ADP'):
            return True

        lemma = t[2]
        if lemma in ('да', 'нет', 'не', 'ни', 'ага'):
            return True

        return False

    def generate_output_by_template(self, output_template, matching, flexer):
        res_words = []
        for word, location, tags in output_template:
            if location is None:
                res_words.append(word)
            else:
                lemma = matching[location]
                all_tags = dict(tags[1:])
                required_tags = ''
                if tags[0] == 'NOUN':
                    required_tags = 'ПАДЕЖ ЧИСЛО'.split()
                elif tags[0] == 'ADJ':
                    required_tags = 'РОД ПАДЕЖ ЧИСЛО ОДУШ СТЕПЕНЬ'.split()
                elif tags[0] == 'VERB':
                    required_tags = 'ВРЕМЯ ЛИЦО ЧИСЛО РОД НАКЛОНЕНИЕ'.split()
                elif tags[0] == 'ADV':
                    required_tags = 'СТЕПЕНЬ'

                required_tags = [(t, all_tags[t]) for t in required_tags if t in all_tags]
                if required_tags:
                    forms = list(flexer.find_forms_by_tags(lemma, required_tags))
                    if forms:
                        form = forms[0]
                    else:
                        form = lemma
                else:
                    form = lemma

                res_words.append(form)

        return ' '.join(res_words)

    def conditional_paraphrase(self, phrase, tags, text_utils):
        # тут пока хардкодим костыли, а в будущем должна быть полномасштабная условная генерация текста...
        if tags[0] == 'same_for_me':
            new_phrase = 'и ' + phrase + ' тоже'
            return new_phrase
        elif tags[0] == 'already_known':
            s0 = random.choice(['я уже знаю, что ', 'мне уже известно, что '])
            new_phrase = s0 + phrase
            return new_phrase
        elif tags[0] == 'opposite_for_me':
            new_phrase = 'а ' + phrase
            return new_phrase
        else:
            raise NotImplementedError()


    def paraphrase(self, phrase, text_utils, bot, session):
        new_phrase = phrase

        if new_phrase in self.processed_phrases:
            phrase_quest = phrase[-1] == '?'
            tokens = text_utils.lemmatize2(new_phrase)
            tokens = [(t[0], t[1].split('|'), t[2]) for t in tokens if self.is_important_token2(t)]

            fx = []
            for i1, (template1, output_templates) in enumerate(self.paraphraser_templates):
                matching = self.match_support_template(template1, tokens, text_utils.word_embeddings)
                if matching:
                    nb_generated = 0
                    for output_template in output_templates:
                        out = self.generate_output_by_template(output_template, matching, text_utils.flexer)
                        out_quest = out[-1] == '?'
                        if out_quest == phrase_quest:
                            #print('{}'.format(out))
                            fx.append(out)

            if fx:
                fx1 = [f for f in fx if f not in self.processed_phrases]
                if len(fx1) == 1:
                    new_phrase = fx1[0]
                elif len(fx1) == 0:
                    new_phrase = random.choice(fx)
                else:
                    new_phrase = random.choice(fx1)
            else:
                if phrase in self.simple_paraphrases:
                    new_phrase = random.choice(self.simple_paraphrases[phrase])

        # Если нам известна гендерная самоидентификация собеседника (см. GenderIdentificationDetector),
        # то может потребоваться скорректировать грамматический род лексики выдаваемой реплики.
        interlocutor_gender = bot.facts.find_tagged_fact(session.get_interlocutor(), '<<<interlocutor_gender>>>')
        if interlocutor_gender:
            if interlocutor_gender == 'ты женского пола':
                interlocutor_gender = 'Fem'
            else:
                interlocutor_gender = 'Masc'

            parsed_data = text_utils.parse_syntax(new_phrase)

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
                        g = text_utils.get_udpipe_attr(pred_token, 'Gender')
                        if g:
                            if g != interlocutor_gender:
                                # нужна коррекция грамматического рода глагола
                                new_verb_form = text_utils.change_verb_gender(pred_token.lemma, interlocutor_gender)
                                if new_verb_form:
                                    new_phrase2 = new_phrase.replace(pred_token.form, new_verb_form)
                                    logging.debug('Changing phrase gender: old_phrase="%s" new_phrase="%s"', new_phrase, new_phrase2)
                                    new_phrase = new_phrase2
                                    break

                if pred_token.head == '0' and pred_token.upos == 'ADJ':
                    for token2 in parsed_data:
                        if token2.head == pred_token.id and token2.upos == 'PRON' and token2.deprel == 'nsubj':
                            if text_utils.get_udpipe_attr(pred_token, 'Gender') != interlocutor_gender:
                                # коррекция для именного сказуемого ("я счастлив" -> "я счастлива") ("ты должен ответить"...)
                                new_adj_form = text_utils.change_adj_gender(pred_token.lemma, interlocutor_gender, text_utils.get_udpipe_attr(pred_token, 'Variant'))
                                if new_adj_form:
                                    new_phrase2 = new_phrase.replace(pred_token.form, new_adj_form)
                                    logging.debug('Changing phrase gender: old_phrase="%s" new_phrase="%s"', new_phrase, new_phrase2)
                                    new_phrase = new_phrase2
                                    break

        self.processed_phrases[new_phrase] += 1

        return new_phrase

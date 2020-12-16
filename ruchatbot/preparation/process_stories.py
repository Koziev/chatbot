"""
Преобразователь образцовых диалогов в наборы правил для вероятностного finite state automata.
Версия 2 - вместо генерации insteadof-rules выдаем особые правила, чтобы нормально работал
выбор перехода с учетом вероятностей.

28-11-2019 Добавлены 2-правила (реакция на реплику собеседника)
14-02-2020 Перенос в основной код бота, упрощение формата записи историй
09-07-2020 Добавлен парсинг содержимого chitchat_stories.txt с мультипликативными сэмплами
11-07-2020 истории A -> B теперь компилируем с условием text, а не raw_text, чтобы использовать результаты
           работы интерпретатора.
"""


import itertools
import collections
import datetime
import os
import io
import re
import pandas as pd
import csv

import tqdm

from ruchatbot.bot.text_utils import TextUtils
from ruchatbot.bot.lgb_synonymy_detector import LGB_SynonymyDetector
from ruchatbot.bot.base_utterance_interpreter2 import BaseUtteranceInterpreter2


BOT_PHRASE = 'B'
HUMAN_PHRASE = 'H'


def is_good_phrase(text):
    return 2 <= len(text) <= 40 and '\\' not in text


def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


class StoryLine:
    def __init__(self):
        self.interrogator = None
        self.text = None

    def __repr__(self):
        return '{}:> {}'.format(self.interrogator, self.text)

    @staticmethod
    def build(text):
        prefixes = [('B:>', BOT_PHRASE), ('B:', BOT_PHRASE), ('B ', BOT_PHRASE),
                    ('H:>', HUMAN_PHRASE), ('H:', HUMAN_PHRASE), ('H ', HUMAN_PHRASE), ]
        for prefix in prefixes:
            if text.startswith(prefix[0]):
                res = StoryLine()
                res.text = text.replace(prefix[0], '').strip()
                res.interrogator = prefix[1]
                return res

        print('Unknown prefix (interrogator marker) in line "{}"'.format(text))
        raise RuntimeError()


class Story:
    def __init__(self):
        self.caption = None
        self.phrases = []

    @staticmethod
    def build(caption, lines):
        story = Story()
        story.caption = caption
        for line in lines:
            if len(line) > 3:
                story.phrases.append(StoryLine.build(line))

        return story


class Stories:
    def __init__(self):
        self.caption2stories = collections.defaultdict(list)

    def count(self):
        return sum(map(len, self.caption2stories.values()))

    def load_toloka(self, toloka_path):
        df = pd.read_csv(toloka_path, encoding='utf-8', delimiter='\t', index_col=None,
                         keep_default_na=False, quoting=csv.QUOTE_MINIMAL)
        for phrases in df['dialogue'].values:
            phrases = phrases.replace('</span>', '</span>\n')
            phrases = phrases.replace('<br />', ' <br />')
            story_lines = []
            current_caption = 'Знакомство'

            phrases = phrases.replace('\n', ' ')
            phrases = phrases.replace('<span class=participant_2>', '\n')
            phrases = phrases.replace('<span class=participant_1>', '\n')

            for phrase in phrases.split('\n'):
                phrase = remove_html_tags(phrase)
                phrase = phrase.replace(u'Пользователь 1:', u'B:> ')
                phrase = phrase.replace(u'Пользователь 2:', u'H:> ')
                phrase = phrase.replace('"', ' ').strip()
                if len(phrase) > 0:
                    story_lines.append(phrase)

            story = Story.build(current_caption, story_lines)
            self.caption2stories[current_caption].append(story)

    def load2(self, chitchat_stories_path):
        print('Loading stories from "{}"...'.format(chitchat_stories_path))
        current_caption = 'болтовня'  # 14-02-2020 считаем, что все истории входят в эту группу по умолчанию
        story_count = 0
        with io.open(chitchat_stories_path, 'r', encoding='utf-8') as rdr:
            lines = []
            for line in rdr:
                s = line.strip()
                if s.startswith('#'):
                    continue

                if s:
                    if s.startswith('-'):
                        s = s[1:].strip()

                    sx = s.split('|')
                    lines.append(sx)
                else:
                    if len(lines) >= 2:
                        for story_lines in itertools.product(*lines):
                            new_lines = []
                            for i, phrase in enumerate(story_lines[::-1]):
                                if 0 == (i % 2):
                                    # Последняя реплика всегда относится к боту
                                    phrase = 'B: ' + phrase
                                else:
                                    # Предпоследняя реплика относится к человеку
                                    phrase = 'H: ' + phrase
                                new_lines.append(phrase)

                            story_lines = new_lines[::-1]
                            story = Story.build(current_caption, story_lines)
                            self.caption2stories[current_caption].append(story)
                            story_count += 1

                    lines = []
        print('{} stories loaded from "{}".'.format(story_count, chitchat_stories_path))

    def load(self, stories_path):
        print('Loading stories from "{}"...'.format(stories_path))
        current_caption = 'болтовня'  # 14-02-2020 считаем, что все истории входят в эту группу по умолчанию
        story_count = 0
        with io.open(stories_path, 'r', encoding='utf8') as rdr:
            story_lines = []
            for iline, line in enumerate(rdr):
                line = line.strip()
                if len(line) == 0:
                    if len(story_lines) > 1:
                        # Используем упрощенный формат.
                        # Каждая реплика истории начинается с "тире".
                        # Расставим метки типа собеседника вместо этих тире.
                        new_lines = []
                        for i, phrase in enumerate(story_lines[::-1]):
                            if phrase.startswith('-'):
                                phrase = phrase[1:].strip()
                                if 0 == (i%2):
                                    # Последняя реплика всегда относится к боту
                                    phrase = 'B: ' + phrase
                                else:
                                    # Предпоследняя реплика относится к человеку
                                    phrase = 'H: ' + phrase
                            new_lines.append(phrase)

                        story_lines = new_lines[::-1]

                        story = Story.build(current_caption, story_lines)
                        story_lines = []
                        self.caption2stories[current_caption].append(story)
                        story_count += 1
                elif line:
                    story_lines.append(line)

            if len(story_lines) == 1:
                print('Single line can not be a story! Near line #{} "{}"'.format(iline, story_lines[0]))
                exit(0)

            if current_caption and len(story_lines):
                story = Story.build(current_caption, story_lines)
                self.caption2stories[current_caption].append(story)
                story_count += 1

        print('{} stories loaded from "{}"'.format(story_count, stories_path))

    def process(self, text_utils, synonymy_detector, word_embeddings):
        caption2builders = dict()
        for caption, stories in self.caption2stories.items():
            # Обрабатываем диалоги в группе с именем caption
            builder = RulesBuilder(caption)
            caption2builders[caption] = builder

            for story in tqdm.tqdm(stories, desc=caption, total=len(stories)):
                if len(story.phrases) == 2:
                    phrase1, phrase2 = story.phrases[0], story.phrases[1]
                    if phrase1.interrogator == HUMAN_PHRASE and\
                        phrase2.interrogator == BOT_PHRASE:
                        if is_good_phrase(phrase1.text) and is_good_phrase(phrase2.text):
                            builder.merge2(text_utils, synonymy_detector, word_embeddings, phrase1.text, phrase2.text)

                for phrase1, phrase2, phrase3 in zip(story.phrases, story.phrases[1:], story.phrases[2:]):
                    if phrase1.interrogator == BOT_PHRASE and\
                       phrase2.interrogator == HUMAN_PHRASE and\
                       phrase3.interrogator == BOT_PHRASE:
                        if is_good_phrase(phrase1.text) and is_good_phrase(phrase2.text) and is_good_phrase(phrase3.text):
                            builder.merge3(text_utils, synonymy_detector, word_embeddings, phrase1.text, phrase2.text, phrase3.text)

        caption2rules = dict((caption, builder.get_rules())
                             for caption, builder
                             in caption2builders.items())
        return caption2rules


def espace_quotes(s):
    return s.replace('"', r'\"')


class Rule2:
    """
    Правило для пар реплик:
    H: Как тебя зовут?
    B: Вика
    """
    def __init__(self, H_phrase):
        self.name = None
        self.H_phrases = [H_phrase]
        self.outputs = set()

    def store_h_phrase(self, H_phrase):
        if H_phrase not in self.H_phrases:
            self.H_phrases.append(H_phrase)

    def __repr__(self):
        return '{} -> {} outputs'.format(self.H_phrase[0], len(self.outputs))

    def store_yaml(self, wrt, person_changer, text_utils):
        wrt.write('    - story_rule:\n')
        wrt.write('        name: "{}"\n'.format(espace_quotes(self.name)))
        wrt.write('        if:\n')

        h_norm = person_changer.flip_person(self.H_phrases[0], text_utils)
        wrt.write('            text: "{}"\n'.format(espace_quotes(h_norm)))

        # В комментариях покажем все опорные фразы
        for h in self.H_phrases[1:]:
            wrt.write('            # "{}"\n'.format(h))

        wrt.write('        then:\n')
        wrt.write('            say:\n')
        for o in self.outputs:
            wrt.write('            - "{}"\n'.format(espace_quotes(o)))
        wrt.write('\n\n')


class Rule3:
    """
    Правило для троек:
    B:> Как тебя зовут?
    H:> Миша
    B:> Приятно познакомиться!
    """
    def __init__(self, B_phrase):
        self.name = None
        self.B_phrases = [B_phrase]
        self.H_phrase2outputs = dict()

    def __repr__(self):
        return '{} -> {} H-entries'.format(self.B_phrase[0], len(self.H_phrase2outputs))

    def store_b_phrase(self, B_phrase):
        if B_phrase not in self.B_phrases:
            self.B_phrases.append(B_phrase)

    def get_H_phrases(self):
        return list(self.H_phrase2outputs.keys())

    def store_yaml(self, wrt, person_changer, text_utils):
        wrt.write('    - story_rule:\n')
        wrt.write('        name: "{}"\n'.format(espace_quotes(self.name)))
        wrt.write('        switch:\n')
        wrt.write('            when:\n')
        wrt.write('                prev_bot_text: "{}"\n'.format(espace_quotes(self.B_phrases[0])))
        for b in self.B_phrases[1:]:
            wrt.write('                # "{}"\n'.format(b))

        wrt.write('            cases:\n')
        for h, outputs in self.H_phrase2outputs.items():
            wrt.write('                - case:\n')
            wrt.write('                    if:\n')
            wrt.write('                        raw_text:\n')
            wrt.write('                            - "{}"\n'.format(espace_quotes(h)))
            wrt.write('                    then:\n')
            wrt.write('                        say:\n')
            for o in outputs:
                wrt.write('                            - "{}"\n'.format(espace_quotes(o)))
            wrt.write('\n')
        wrt.write('\n\n')


class RulesBuilder:
    def __init__(self, caption):
        self.caption = caption
        self.rules2 = []
        self.rules3 = []

    def get_B_phrases(self):
        return [rule.B_phrases[0] for rule in self.rules3]

    def get_H_phrases(self):
        return [rule.H_phrases[0] for rule in self.rules2]

    def merge2(self, text_utils, synonymy_detector, word_embeddings, phrase1, phrase2):
        # Найдем ближайшую первую фразу (H)
        phrase1_x = text_utils.wordize_text(phrase1)
        hx = [(text_utils.wordize_text(b), None, None) for b in self.get_H_phrases()]

        rule2 = None

        if len(hx) > 0:
            best_h, best_sim = synonymy_detector.get_most_similar(phrase1_x,
                                                                  hx,
                                                                  text_utils,
                                                                  nb_results=1)

            if best_sim < 0.80:
                # Ничего похожено на phrase1 еще нет, создадим новый якорь.
                rule2 = Rule2(phrase1_x)
                self.rules2.append(rule2)
            else:
                # Нашли фразу с похожей опорной фразой.
                h_index = next(i for (i, h) in enumerate(hx) if h[0] == best_h)
                rule2 = self.rules2[h_index]
                rule2.store_h_phrase(phrase1_x)

        if rule2 is None:
            rule2 = Rule2(phrase1_x)
            self.rules2.append(rule2)

        # Добавляем новую выдаваемую ботом реплику
        rule2.outputs.add(phrase2)

    def merge3(self, text_utils, synonymy_detector, word_embeddings, phrase1, phrase2, phrase3):
        # Найдем ближайшую первую фразу (B)
        phrase1_x = text_utils.wordize_text(phrase1)
        bx = [(text_utils.wordize_text(b), None, None) for b in self.get_B_phrases()]

        rule3 = None

        if len(bx) > 0:
            best_b, best_sim = synonymy_detector.get_most_similar(phrase1_x,
                                                                  bx,
                                                                  text_utils,
                                                                  nb_results=1)

            if best_sim < 0.80:
                # Ничего похожено на phrase1 еще нет, создадим новый якорь.
                rule3 = Rule3(phrase1_x)
                self.rules3.append(rule3)
            else:
                b_index = next(i for (i, b) in enumerate(bx) if b[0] == best_b)
                rule3 = self.rules3[b_index]

        if rule3 is None:
            rule3 = Rule3(phrase1_x)
            self.rules3.append(rule3)
        else:
            rule3.store_b_phrase(phrase1_x)

        # Теперь ищем в rule3 ближайший второй шаг для H-фразы
        hx = [(text_utils.wordize_text(h), None, None) for h in rule3.get_H_phrases()]
        phrase2_x = text_utils.wordize_text(phrase2)

        h_entry = None
        if len(hx) > 0:
            best_h, best_sim = synonymy_detector.get_most_similar(phrase2_x,
                                                                  hx,
                                                                  text_utils,
                                                                  nb_results=1)

            if best_sim > 0.80:
                h_index = next(i for (i, h) in enumerate(hx) if h[0] == best_h)
                h_entry = rule3.H_phrase2outputs[rule3.get_H_phrases()[h_index]]

        if h_entry:
            # К имеющейся H-фразе добавляем новый вариант выдаваемой реплики
            if phrase3 not in h_entry:
                h_entry.add(phrase3)
        else:
            # Добавляем новую H-фразу и выдаваемую ботом реплику
            rule3.H_phrase2outputs[phrase2_x] = set([phrase3])

    def get_rules2(self):
        return self.rules2

    def get_rules3(self):
        return self.rules3

    def get_rules(self):
        return self.rules3 + self.rules2


if __name__ == '__main__':
    models_folder = '/home/inkoziev/polygon/chatbot/tmp'
    data_folder = '/home/inkoziev/polygon/chatbot/data'
    stories_file = os.path.join(data_folder, 'stories.txt')
    chitchat_stories_file = os.path.join(data_folder, 'chitchat_stories.txt')
    TOLOKA = False

    if TOLOKA:
        output_file = '/home/inkoziev/polygon/chatbot/tmp/toloka_generated_rules.yaml'
    else:
        output_file = '/home/inkoziev/polygon/chatbot/tmp/generated_rules.yaml'

    text_utils = TextUtils()
    text_utils.load_dictionaries(data_folder, models_folder)

    synonymy_detector = LGB_SynonymyDetector()
    synonymy_detector.load(models_folder)

    person_changer = BaseUtteranceInterpreter2()
    person_changer.load(models_folder)
    # НАЧАЛО ОТЛАДКИ
    #sss = person_changer.flip_person('я люблю компьютерные игры', text_utils)
    # КОНЕЦ ОТЛАДКИ

    word_embeddings = None
    stories = Stories()

    if TOLOKA:
        stories.load_toloka('/home/inkoziev/polygon/chatbot/data/TolokaDialogues/TlkPersonaChatRus/TlkPersonaChatRus/dialogues.tsv')
    else:
        stories.load(stories_file)
        stories.load2(chitchat_stories_file)

    print('Compilation of rules...')
    caption2rules = stories.process(text_utils, synonymy_detector, word_embeddings)

    with io.open(output_file, 'w', encoding='utf8') as wrt:
        wrt.write('# {} chitchat rules compiled {}\n'.format(stories.count(), datetime.datetime.now()))
        wrt.write('story_rules:\n')
        for caption, rules in caption2rules.items():
            for i, rule in enumerate(rules):
                rule.name = 'rule_builder[' + caption + '_' + str(i)+']'
                rule.store_yaml(wrt, person_changer, text_utils)

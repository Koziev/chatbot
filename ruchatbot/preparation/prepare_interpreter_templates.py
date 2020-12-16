# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки модели интерпретации ответов в чатботе. Эта модель
восстанавливает полный (подразумеваемый, развернутый) текст ответа по фактическому
краткому ответу собеседника, используя контекст в виде заданного вопроса. Обрабатывается анафора,
гэпинг, эллипсис.

28-03-2020 Добавлен отчет по OOV словам в датасете
30-05-2020 Добавлена загрузка сэмплов из question interpretation датасета в общую кучу
20-06-2020 Добавлена генерация шаблонов интерпретатора (knn1-модель)
10-07-2020 Ансамбль тэггеров для увеличения точности тегирования существительных
02-08-2020 Эксперимент с добавлением 1 или 2 контекстов для каждого сэмпла, чтобы модель сама научилась
           выбирать релевантные контексты. Также убираем подготовку для старой модели интерпретатора (генеративной).
24-11-2020 Добавлено формирование датасета interpreter_samples_for_pretrain.tsv с неотсмотренными (автосгенерированными)
           сэмплами для претренировки.
29-11-2020 Добавлен поиск невалидных сэмплов в interpretation.txt: где первая фраза контекста и раскрытая реплика обе
           в 1м лице
"""

import io
import collections
import os
import re
import pickle
import itertools
import random

import pandas as pd
import pyconll

from ufal.udpipe import Model, Pipeline, ProcessingError
from rnnmorph.predictor import RNNMorphPredictor

import rutokenizer
import rupostagger
import ruword2tags
import rulemma


def is_int(s):
    return re.match(r'^\d+$', s)


class TaggerEnsemble:
    def __init__(self):
        self.predictor = RNNMorphPredictor(language="ru")

        self.tagger = rupostagger.RuPosTagger()
        self.tagger.load()

        #model_file = '/home/inkoziev/polygon/GramEval2020/tmp/udpipe_syntagrus.model'
        #self.ud_model = Model.load(model_file)
        #self.ud_pipeline = Pipeline(self.ud_model, 'vertical', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        #self.ud_error = ProcessingError()

    def tag(self, words):
        tokens1 = self.tagger.tag(words)
        tokens2 = self.predictor.predict(words)

        #processed = self.ud_pipeline.process('\n'.join(words), self.ud_error)
        #if self.ud_error.occurred():
        #    print("An error occurred when running run_udpipe: ")
        #    print(self.ud_error.message)
        #    return tokens1
        #tokens3 = pyconll.load_from_string(processed)[0]

        new_tokens = []
        for token1, token2 in zip(tokens1, tokens2):
            tags1 = token1[1].split('|')
            if tags1[0] == 'NOUN' and 'Case' in token2.tag:
                tags_rnn = dict(z.split('=') for z in token2.tag.split('|') if '=' in z)
                new_tagset = list(filter(lambda z: not z.startswith('Case'), tags1))
                new_tagset.append(('Case='+tags_rnn['Case']))
                new_tokens.append((token1[0], '|'.join(new_tagset)))
            else:
                new_tokens.append(token1)

        return new_tokens


class Sample2:
    def __init__(self, left, short_phrase, expanded_phrase, handcrafted=False):
        self.handcrafted = handcrafted
        self.left = left
        self.short_phrase = short_phrase
        self.expanded_phrase = expanded_phrase


def wx(s):
    return ' '.join(tokenizer.tokenize(s)).lower()


def is_important_token2(pos, lemma):
    if pos in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'ADP', 'PREP'):
        return True

    if lemma in ('да', 'нет', 'не', 'ни', 'ага'):
        return True

    return False


salient_tags = 'Case Gender Number Tense Person Mood Degree'.split()


class ContextTemplateItem:
    def __init__(self, form, lemma, tags, location):
        self.form = form
        self.lemma = lemma
        self.tags = tags
        self.location = location

    def __repr__(self):
        s = self.form
        if self.tags:
            s += '(' + '|'.join(self.tags) + ')'
            s += '@{};{}'.format(self.location[0], self.location[1])
        return s


class ContextTemplateLine:
    def __init__(self, source_str, items):
        self.source_str = source_str
        self.items = items

    def __repr__(self):
        return self.source_str

    def generate_packed_template(self):
        res = []
        for item in self.items:
            if item.tags:
                res.append((None, item.tags, item.location))
            else:
                res.append((item.form, None, None))
        return tuple(res)


def create_context_template(context_line_index, source_str, expanded_tokens):
    items = []
    expanded_lemmas = [t[2] for t in expanded_tokens]

    tokens = lemmatizer.lemmatize(tagger.tag(tokenizer.tokenize(source_str)))

    for token_index, token in enumerate(tokens):
        lemma = token[2]
        tags = token[1].split('|')
        pos = tags[0]
        if is_important_token2(pos, lemma) or len(tokens) == 1:
            item = None
            if pos in ('NOUN', 'ADJ', 'VERB', 'ADV', 'NUM'):
                if lemma in expanded_lemmas:
                    # Это слово уходит в результат
                    # Оставим только проверку важных тэгов
                    tags = tuple([pos] + list(filter(lambda t: t.split('=')[0] in salient_tags, tags)))
                    item = ContextTemplateItem(token[0], lemma, tags, (context_line_index, token_index))

            if item is None:
                item = ContextTemplateItem(token[0], lemma, None, None)

            items.append(item)

    if len(items) == 0:
        print('ERROR@225: empty template generated for "{}"'.format(' '.join(t[0] for t in tokens)))
        return None

    return ContextTemplateLine(source_str, items)


def create_expanded_template(context_templates, expanded_tokens):
    items = []

    for token in expanded_tokens:
        lemma = token[2]
        tags = token[1]
        item2 = None
        # Ищем, откуда из контекста можно скопировать эту лемму
        for template in context_templates:
            for item_index, item in enumerate(template.items):
                if item.location is not None and item.lemma and item.lemma == lemma:
                    out_tags = [tags.split('|')[0]]
                    for tag in tags.split('|')[1:]:
                        tname, tval = tag.split('=')
                        if tname in salient_tags:
                            if tname == 'Number':
                                if tval == 'Sing':
                                    out_tags.append(('ЧИСЛО', 'ЕД'))
                                else:
                                    out_tags.append(('ЧИСЛО', 'МН'))

                            elif tname == 'Degree':
                                if tval == 'Cmp':
                                    out_tags.append(('СТЕПЕНЬ', 'СРАВН'))
                                elif tval == 'Sup':
                                    out_tags.append(('СТЕПЕНЬ', 'ПРЕВОСХ'))
                                else:
                                    out_tags.append(('СТЕПЕНЬ', 'АТРИБ'))

                            elif tname == 'Case':
                                if tval == 'Nom':
                                    out_tags.append(('ПАДЕЖ', 'ИМ'))
                                elif tval == 'Gen':
                                    out_tags.append(('ПАДЕЖ', 'РОД'))
                                elif tval == 'Ins':
                                    out_tags.append(('ПАДЕЖ', 'ТВОР'))
                                elif tval == 'Acc':
                                    out_tags.append(('ПАДЕЖ', 'ВИН'))
                                elif tval == 'Dat':
                                    out_tags.append(('ПАДЕЖ', 'ДАТ'))
                                elif tval == 'Loc':
                                    out_tags.append(('ПАДЕЖ', 'ПРЕДЛ'))
                            elif tname == 'Gender':
                                if tval == 'Fem':
                                    out_tags.append(('РОД', 'ЖЕН'))
                                elif tval == 'Masc':
                                    out_tags.append(('РОД', 'МУЖ'))
                                else:
                                    out_tags.append(('РОД', 'СР'))
                            elif tname == 'Tense':
                                if tval == 'Past':
                                    out_tags.append(('ВРЕМЯ', 'ПРОШЕДШЕЕ'))
                                elif tval == 'Pres':
                                    out_tags.append(('ВРЕМЯ', 'НАСТОЯЩЕЕ'))
                                else:
                                    out_tags.append(('ВРЕМЯ', 'БУДУЩЕЕ'))
                            elif tname == 'Person':
                                if tval == '1':
                                    out_tags.append(('ЛИЦО', '1'))
                                elif tval == '2':
                                    out_tags.append(('ЛИЦО', '2'))
                                else:
                                    out_tags.append(('ЛИЦО', '3'))
                            elif tname == 'Mood':
                                if tval == 'Imp':
                                    out_tags.append(('НАКЛОНЕНИЕ', 'ПОБУД'))
                                elif tval == 'Ind':
                                    out_tags.append(('НАКЛОНЕНИЕ', 'ИЗЪЯВ'))

                    item2 = (None, item.location, tuple(out_tags))
                    break

            if item2:
                break

        if item2 is None:
            item2 = (token[0], None, None)

        items.append(item2)

    return tuple(items)


if __name__ == '__main__':
    data_dir = '../../data'
    tmp_dir = '../../tmp'

    augment_with_random_phrases = True

    input_dataset = os.path.join(data_dir, 'interpretation.txt')
    input_dataset2 = os.path.join(data_dir, 'question_interpretation.txt')
    input_dataset3 = os.path.join(data_dir, 'interpretations2.txt')

    raw_input_datasets = [os.path.join(tmp_dir, 'raw_interpretations_dataset.4.txt'),
                          os.path.join(tmp_dir, 'answer_interpretation.raw.txt'),
                          os.path.join(data_dir, 'interpretation_auto_4.txt'),
                          os.path.join(data_dir, 'interpretation_auto_5.txt'),
                          os.path.join(data_dir, 'interpretation_auto_neg4.txt'),
                          ]

    output_dataset2 = os.path.join(tmp_dir, 'interpreter_samples.tsv')
    raw_output_dataset2 = os.path.join(tmp_dir, 'interpreter_samples_for_pretrain.tsv')

    rare_path = os.path.join(tmp_dir, 'most_rare_interpretaions.txt')
    no_expansion_path = os.path.join(tmp_dir, 'interpretation_no_expansion_phrases.txt')

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    # Поиск невалидных сэмплов, в которых грамматическое лицо раскрытой фразы совпадает с лицом контекста
    print('Searching for bad samples in "{}"...'.format(input_dataset))
    with io.open(os.path.join(tmp_dir, 'interpretation.bad_samples.txt'), 'w', encoding='utf-8') as wrt_bad,\
         io.open(input_dataset, 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if line:
                lines.append(line)
            else:
                if len(lines) == 2:
                    question = wx(lines[0])
                    if '|' in lines[1]:
                        tx = lines[1].split('|')
                        short_phrase = wx(tx[0].strip())
                        expanded_phrase = wx(tx[1].strip())

                        context_tx = question.split()
                        expanded_tx = expanded_phrase.split()
                        is_bad = False
                        for keyword in 'мне я ты тебе мной меня'.split():
                            if keyword in context_tx and keyword in expanded_tx:
                                is_bad = True
                                break

                        if is_bad:
                            wrt_bad.write('{}\n{}\n\n\n'.format(lines[0], lines[1]))


    # Соберем рандомные фразы для набивки контекста
    random_phrases = set()

    # реплики из тестового бота
    with io.open(os.path.join(data_dir, 'test', 'test_phrases.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            line = line.strip()
            if line and not line.startswith('#'):
                phrase = line
                random_phrases.add(phrase)

    with io.open(os.path.join(data_dir, 'test', 'test_dialogues.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            line = line.strip()
            if line and not line.startswith('#') and '~' not in line and line not in ('BH', 'HBH'):
                line = line.replace('B:', '').replace('H:', '').strip()
                for s in line.split('|'):
                    phrase = s.strip()
                    random_phrases.add(phrase)

    with io.open(os.path.join(tmp_dir, 'pqa_all.dat'), 'r') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if line:
                lines.append(line.strip())
            else:
                a = lines[-1]
                q = lines[-2] + ' ?'
                random_phrases.add(a)
                random_phrases.add(q)
                lines = []

    fx = ['понятно', 'Я поняла)', 'Понятно))', 'Понятненько.', 'Понятно',
          'Я понял', 'да, я понял', 'спасибо, понял', 'спасибо за пояснения, понял',
          'мне все понятно', 'вот теперь понятно', 'ага, понял', 'мысль понятна',
          'не, ну понял я', 'теперь понятно', 'я понял', 'да поняла я, поняла',
          'ага, понял', 'Что тебя интересует?', 'что ты имеешь в виду?',
          'ничего не понял', 'к сожалению, не понял', 'неа, не понял',
          'не понятно мне', 'вообще ничего не понял', 'не уловил мысль',
          'снова не понятно', 'не понял', 'я не понял', 'не понял тебя',
          'я это не понял', 'я не поняла это', 'что-то нихрена не понял...',
          'Мне очень жаль.', 'жаль', 'печаль', 'печалька', 'печально',
          'В общем я очень сильно огорчён', 'это печально', 'эх, жаль',
          'ну что ж, жалко', 'жалко, конечно', 'печально', 'это печально',
          'Я очень сожалею.', 'Вот ведь невезуха...', 'Очень жаль(', 'Эх',
          'печалька', 'это печально', 'мне кажется, это печально', 'как жаль',
          'жалко-то как', 'эх, жалко', ':(', 'ух ты, ух ты', 'ах вот как',
          'ах вот оно что...', 'ух ты', 'вот нифига ж себе', 'рукалицо',
          'вау, круто', 'вау!', 'вау', 'хм', 'o_O', 'O_o', ':-[', 'ты серьезно?',
          'о как', 'Мне жаль, что ты расстроен.', 'мда', 'пф', 'опаньки', 'ой',
          'фух', 'ого', 'ого, круто', 'ну прям не знаю', 'ну не знаю, не знаю',
          'ух ты', 'ничего себе', 'фига себе', 'ни фига себе', 'вот это да',
          'фига ж себе', 'ни хуя себе', 'полный отпад', 'вот это да', 'ммм',
          'хм', 'вот те раз', 'ну и дела', 'уф', 'во дела', 'ни фига себе',
          'отлично', 'прелестно', 'Ты облажалась.', 'Твою мать!', 'Опаньки.',
          'Благодарю за ответ', 'Спасибо за ответ!', 'Рада знакомству!',
          'Ну елки!', 'Елки-палки!', 'Елки зеленые!', 'Елки ж палки!',
          'До чего дошел прогресс!', 'ты в своём уме?', 'О, простите, сэр!',
          'Я это знаю, знаю!', '- Я так и думала.', '- Ну, не знаю ...',
          'Крепись, дружище.', 'Вот умора!', 'Не будь дураком.',
          'Не будь глупцом!', 'Не будь циником.', 'Ну и умора!',
          'А мне это без разницы', 'Мне это не интересно',
          'Да наплевать мне на него', 'Наплевать мне на это',
          'Наплевать на них, вот и все!', 'Ну вот, наконец-то!',
          'Мне не интересно об этом говорить', 'Нет, понятно, понятно',
          'Что же ты....', 'Да я что...', 'Батюшки!', 'Мама родная!',
          'С ума сойти!', 'Вот еще!', 'Ну ладно', 'Да что ты говоришь!',
          'Ну ты даешь!', 'Ну и ну', 'Вот еще!', 'Честное слово.', 'Чес слово',
          'нет , не надо извиняться', 'Вот и замечательно!', 'Вау! Поздравляю!!',
          'было дело', 'ты тут как тут', 'Это я любя', 'это не к добру',
          'эй, поаккуратнее!', 'простите', 'браво!', 'просто супер!',
          'не извиняйтесь', 'некогда извиняться', 'ага, увидимся',
          'не расстраивайся', 'в любом случае, мне плевать', 'говори за себя',
          'извини, пожалуйста', 'извини', 'извини меня', 'извини, пожалуйста',
          'прости', 'простите меня', 'ой', 'ой, извините меня', 'о, зашибись',
          'да это зашибись', 'это офигенно', 'прости меня, пожалуйста',
          'да ничего', 'ладно', 'мне жаль', 'и мне тоже жаль', 'Да... мечты)',
          'Ну и отлично', 'Ух ты', 'это фигово', 'Сорри', 'не волнуйся',
          'блин', 'Извини', 'Эхе-хе-нюшки', 'пожалуйста не спрашивай',
          'Завидую белой завистью!', 'Звучит отлично, спасибо.',
          'Приятно было пообщаться', 'Хорошо, отлично. Спасибо! ',
          'У меня всё в порядке.', 'Благодарю за вечер!', 'Везет тебе)',
          'черт, никогда бы не подумал', 'Отлично, спасибо, что помогли мне',
          'Хорошо, спасибо, что нашли это для меня',
          'Хорошо, я думаю, это полезно для меня',
          'О, замечательно. Спасибо за информацию.',
          'О, это звучит потрясающе. Огромное спасибо.', 'как интересно :)',
          'Приятного))', 'Ой блин', 'Давай, удачи))', 'Типа того)',
          'Ясно', 'взаимно)', 'отличный выбор профессии!', 'Ух ты)',
          'Рад новому знакомству!', 'Супер, поздравляю!', 'Вот в чём дело',
          'очень приятно)', 'аллергия-это плохо(((', 'я прям тоже самое писала!',
          'Ого, это сложно', 'Круто:)', 'ахахах', 'Интересно)', 'оу)ахахахах',
          'Приветик, ничего себе)))', 'Приятно познакомится', 'Ахах)',
          'Спасибо за комплимент ))))', 'Это здорово )))', 'Это точно)',
          'О да круто', 'Чтооо ((((', 'Ох, соболезную ..',
          'Я поднимаю брови, показывая любопытство.', 'Взаимно',
          'Я оглядел ресторан, чувствуя неловкость.', 'как-то печально это',
          'Я улыбнулся, заранее ожидая комплименты.', 'ого',
          'Хаха', 'вау', 'Рада слышать это, дорогой', 'Боже праведный',
          'Видите ли... ', 'Какой чудесный день!', 'ну ничего себе',
          'Конечно,понимаю!', 'Оооо спорт это сила', 'вот это да',
          'не может быть!', 'обалдеть', 'офигеть', 'Что вы, не стоит.',
          'Пожалуйста!', 'О боже.', 'Слава богу!', 'Хвала Аллаху!',
          'Боже правый!', 'Пустяки.', 'Ерунда.', 'Это мелочи.', 'Вот черт.',
          'Все получится!', 'Ха!', 'Ой', 'Вот те на!', 'Да ну блин', 'Блин',
          'Елки-палки', 'да елки ж палки', 'ну и ну', 'ух, елки',
          'вот тебе бабушка и юрьев день', 'ну ничего себе', 'О Господи!',
          'Извини.', 'Извините меня ради бога', 'Извини меня', 'Прошу прощения!',
          'Странно это все', 'Это странно', 'ну что ж', 'Ну что же...',
          'звучит странно', 'звучит разумно', 'извини', 'да ну как же это',
          'как-то это странно', 'не знаю, как к этому отнестись',
          'мне плевать на это', 'да насрать мне на это', 'ужас-то какой',
          'неплохо, неплохо', 'мммм', 'как же я ненавижу все это',
          'мне жаль тебя', 'я рад за тебя', 'приятно это слышать',
          'приятно слышать', 'а ты знаешь, я рад за это', 'а я доволен этим',
          'нефиговенько так', 'однако ж', 'однако!', 'неожиданно',
          'не знаю, что и сказать', 'прямо не знаю, как ответить',
          'затрудняюсь с ответом', 'мне нечего сказать',
          'обсуждали уже это', 'мы об этом уже говорили']
    random_phrases.update(fx)

    if False:
        for fname in ['questions.txt', 'premises.txt']:
            with io.open(os.path.join(tmp_dir, fname), 'r', encoding='utf-8') as rdr:
                for line in rdr:
                    s = line.strip()
                    if s:
                        random_phrases.add(wx(s))

    random_phrases = [wx(s) for s in random_phrases]

    samples2 = []
    no_expansion_phrases = []
    nb_samples1 = 0
    print('Processing "{}"...'.format(input_dataset))
    with io.open(input_dataset, 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if line:
                lines.append(line)
            else:
                if len(lines) == 2:
                    question = wx(lines[0])
                    if '|' in lines[1]:
                        tx = lines[1].split('|')
                        if len(tx) > 2:
                            print('Invalid data format in line "{}"'.format(lines[1]))
                            exit(0)

                        short_phrase = wx(tx[0].strip())
                        expanded_phrase = wx(tx[1].strip())
                        if len(expanded_phrase) < 2:
                            print('Empty expanded answer for question="{}", short_phrase="{}"'.format(question, short_phrase))

                        sample2 = Sample2(question, short_phrase, expanded_phrase, handcrafted=True)
                        samples2.append(sample2)
                        nb_samples1 += 1

                        if augment_with_random_phrases:
                            # Добавляем 1 рандомную фразу в кач-ве первого контекста
                            context = [random.choice(random_phrases), question]
                            left = ' | '.join(context)
                            sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=False)
                            samples2.append(sample2)

                elif len(lines) == 1:
                    tx = lines[0].split('|')
                    if len(tx) == 1:
                        phrase1 = tx[0].strip()
                        no_expansion_phrases.append(u' '.join(tokenizer.tokenize(phrase1)))
                    elif len(tx) == 2:
                        phrase1 = tx[0].strip()
                        phrase2 = tx[1].strip()
                        if len(phrase2) == 0 or phrase1 == phrase2:
                            no_expansion_phrases.append(u' '.join(tokenizer.tokenize(phrase1)))

                        # Сэмпл без контекста
                        short_phrase = wx(tx[0].strip())
                        expanded_phrase = wx(tx[1].strip())
                        sample2 = Sample2('', short_phrase, expanded_phrase, handcrafted=True)
                        samples2.append(sample2)
                        nb_samples1 += 1

                        if augment_with_random_phrases:
                            # добавим 1 рандомный контекст
                            context = [random.choice(random_phrases)]
                            left = ' | '.join(context)
                            sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=False)
                            samples2.append(sample2)

                            # добавляем 2 рандомных контекста
                            context = [random.choice(random_phrases), random.choice(random_phrases)]
                            left = ' | '.join(context)
                            sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=False)
                            samples2.append(sample2)
                elif len(lines) >= 3:
                    question = wx(lines[0])
                    lines2 = [wx(s.strip()) for s in lines[1].split('|')]
                    lines3 = [wx(s.strip()) for s in lines[2].split('|')]

                    if len(lines2) == 2:
                        short_phrase = lines2[0]
                        expanded_phrase = lines2[1]
                        sample2 = Sample2(question, short_phrase, expanded_phrase, handcrafted=True)
                        samples2.append(sample2)
                        nb_samples1 += 1

                        if augment_with_random_phrases:
                            # Добавляем 1 рандомную фразу в кач-ве первого контекста
                            context = [random.choice(random_phrases), question]
                            left = ' | '.join(context)
                            sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=False)
                            samples2.append(sample2)

                    if len(lines3) == 2:
                        context = [question, lines2[0]]
                        left = ' | '.join(context)
                        short_phrase = lines3[0]
                        expanded_phrase = lines3[1]
                        sample2 = Sample2(left, short_phrase, expanded_phrase)
                        samples2.append(sample2)
                        nb_samples1 += 1

                lines = []

    print('{} samples loaded from "{}"'.format(nb_samples1, input_dataset))

    nb_samples2 = 0
    print('Processing "{}"...'.format(input_dataset2))
    with io.open(input_dataset2, 'r', encoding='utf-8') as rdr:
        lines = []
        for iline, line in enumerate(rdr):
            line = line.strip()
            if line:
                lines.append(line)
            else:
                if len(lines) == 3:
                    question = lines[0]
                    answer = lines[1]

                    tx = lines[-1].split('|')
                    if len(tx) != 2:
                        print('Corrupted data format in line {} "{}" file="{}"'.format(iline, lines[-1], input_dataset2))
                        exit(0)

                    short_phrase = tx[0].strip()
                    expanded_phrase = tx[1].strip()

                    if len(expanded_phrase) < 2:
                        print('Empty expanded phrase for short_phrase="{}" in line {}'.format(short_phrase), iline)
                        exit(0)

                    left = wx(' | '.join([wx(s) for s in lines[:-1]]))
                    short_phrase = wx(short_phrase)
                    expanded_phrase = wx(expanded_phrase)

                    sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=True)
                    samples2.append(sample2)
                    nb_samples2 += 1

                lines = []

    print('{} samples loaded from "{}"'.format(nb_samples2, input_dataset2))

    # сэмплы из этого датасета берем без добавления рандомных фраз в контекст
    print('Processing "{}"...'.format(input_dataset3))
    nb_samples3 = 0
    with io.open(input_dataset3, 'r', encoding='utf-8') as rdr:
        lines = []
        for iline, line in enumerate(rdr):
            line = line.strip()
            if line.startswith('#'):
                continue
            elif line:
                lines.append(line)
            else:
                if len(lines) > 0:
                    tx = lines[-1].split('|')
                    if len(tx) != 2:
                        print('Corrupted data format in line {} "{}" file="{}"'.format(iline, lines[-1], input_dataset3))
                        exit(0)

                    short_phrase = tx[0].strip()
                    expanded_phrase = tx[1].strip()

                    if len(expanded_phrase) < 2:
                        print('Empty expanded phrase for short_phrase="{}" in line {}'.format(short_phrase), iline)
                        exit(0)

                    left = wx(' | '.join([wx(s) for s in lines[:-1]]))
                    short_phrase = wx(short_phrase)
                    expanded_phrase = wx(expanded_phrase)

                    sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=True)
                    samples2.append(sample2)
                    nb_samples3 += 1

                lines = []

    print('{} samples loaded from "{}"'.format(nb_samples3, input_dataset3))

    nb_samples = nb_samples1 + nb_samples2 + nb_samples3
    print('{} samples loaded in total (not counting augmented ones)'.format(nb_samples))

    print('Loading raw samples for pre-training...')
    raw_samples2 = []
    for p in raw_input_datasets:
        print('Processing raw samples from "{}"...'.format(p))
        with io.open(p, 'r', encoding='utf-8') as rdr:
            lines = []
            for iline, line in enumerate(rdr):
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif line:
                    lines.append(line)
                else:
                    if len(lines) > 0:
                        tx = lines[-1].split('|')
                        if len(tx) != 2:
                            print('Corrupted data format in line {} "{}" file="{}"'.format(iline, lines[-1], p))
                            continue

                        short_phrase = tx[0].strip()
                        expanded_phrase = tx[1].strip()

                        if len(expanded_phrase) < 2:
                            print('Empty expanded phrase for short_phrase="{}" in line {}'.format(short_phrase), iline)
                            exit(0)

                        left = wx(' | '.join([wx(s) for s in lines[:-1]]))
                        short_phrase = wx(short_phrase)
                        expanded_phrase = wx(expanded_phrase)

                        sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=True)
                        raw_samples2.append(sample2)

                    lines = []

    print('{} raw samples loaded'.format(len(raw_samples2)))


    # Добавляем в датасет также фразы, которые не надо интерпретировать.
    iden_phrases = set()

    if False:
        # Из таблицы трансляции приказов возьмем все строки - они не требуют интерпретации
        with io.open(os.path.join(data_dir, 'orders.txt'), 'r', encoding='utf-8') as rdr:
            for line in rdr:
                s = line.strip()
                if s:
                    iden_phrases.add(s)

    # Из демо-FAQ возьмем вопросы, они тоже гарантированно полные
    with io.open(os.path.join(data_dir, 'faq2.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s and s.startswith(u'Q: '):
                s = s.replace(u'Q:', u'').strip()
                iden_phrases.add(s)

    # Из PQA датасета берем предпосылки и вопросы
    df = pd.read_csv(os.path.join(tmp_dir, 'premise_question_relevancy.csv'),
                     encoding='utf-8',
                     delimiter='\t',
                     quoting=3)

    for premise in set(df.premise.values[:10000]):
        iden_phrases.add(premise)

    for question in set(df.question.values[:10000]):
        iden_phrases.add(question)

    if augment_with_random_phrases:
        for phrase in iden_phrases:
            short_phrase = phrase
            expanded_phrase = phrase

            # добавим 1 рандомный контекст
            context = [random.choice(random_phrases)]
            left = wx(' | '.join(context))
            sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=False)
            samples2.append(sample2)

            # добавляем 2 рандомных контекста
            context = [random.choice(random_phrases), random.choice(random_phrases)]
            left = wx(' | '.join(context))
            sample2 = Sample2(left, short_phrase, expanded_phrase, handcrafted=False)
            samples2.append(sample2)

    print('{} samples, {} no-expansion phrases'.format(len(samples2), len(no_expansion_phrases)))

    print('Writing {} samples to "{}"'.format(len(samples2), output_dataset2))
    output2freq = collections.Counter()
    with io.open(output_dataset2, 'w', encoding='utf-8') as wrt:
        wrt.write('context\toutput\thandmade\n')
        for sample in samples2:
            if sample.left:
                context = sample.left + ' | ' + sample.short_phrase
            else:
                context = sample.short_phrase

            output = sample.expanded_phrase
            wrt.write('{}\t{}\t{}\n'.format(context, output, sample.handcrafted))
            output2freq[output] += 1

    # В отдельном файле сохраним сырые сэмплы для претренировки.
    # Чтобы валидационные сэмплы не попали в тренировку, не будем добавлять в эту кучу чистые сэмплы.
    print('Storing {} raw samples for pre-training...'.format(len(raw_samples2)))
    with io.open(raw_output_dataset2, 'w', encoding='utf-8') as wrt:
        wrt.write('context\toutput\thandmade\n')
        for sample in raw_samples2:
            if sample.left:
                context = sample.left + ' | ' + sample.short_phrase
            else:
                context = sample.short_phrase

            output = sample.expanded_phrase
            wrt.write('{}\t{}\t{}\n'.format(context, output, sample.handcrafted))


    # Сформируем отчет о самых редких интерпретациях, чтобы можно было аугментировать их руками и нарастить частоту.
    with io.open(rare_path, 'w', encoding='utf-8') as wrt:
        rare_outputs = set(output for output, freq in output2freq.items() if freq == 1)
        for sample in sorted(samples2, key=lambda s: (len(s.short_phrase), len(s.left))):
            if sample.expanded_phrase in rare_outputs:
                for context_phrase in sample.left.split('|'):
                    wrt.write('{}\n'.format(context_phrase.strip()))
                wrt.write('{} | {}\n\n\n'.format(sample.short_phrase, sample.expanded_phrase))

    print('Writing {} samples to "{}"'.format(len(no_expansion_phrases), no_expansion_path))
    with io.open(no_expansion_path, 'w', encoding='utf=8') as wrt:
        for phrase in no_expansion_phrases:
            wrt.write(phrase+'\n')

    #tagger = rupostagger.RuPosTagger()
    #tagger.load()

    tagger = TaggerEnsemble()

    # НАЧАЛО ОТЛАДКИ
    #tagsets = tagger.tag('меня зовут илья'.split())
    # КОНЕЦ ОТЛАДКИ

    gren = ruword2tags.RuWord2Tags()
    gren.load()

    # НАЧАЛО ОТЛАДКИ
    #words = tokenizer.tokenize('Тебе нравится пить кофе')
    #tags = list(tagger.tag(words))
    # КОНЕЦ ОТЛАДКИ

    lemmatizer = rulemma.Lemmatizer()
    lemmatizer.load()

    # Поищем несловарные и нечисловые токены
    vocabulary = set()
    with io.open(os.path.join(data_dir, 'dict/word2lemma.dat'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            fields = line.strip().split('\t')
            if len(fields) == 4:
                word = fields[0].lower().replace(' - ', '-')
                vocabulary.add(word)

    oov_tokens = set()

    # Делаем морфологические шаблоны для модели интерпретации knn-1
    print('Building knn1 templates from {} samples...'.format(len(samples2)))
    templates2 = collections.Counter()
    packed2samples = collections.defaultdict(list)
    for sample in samples2:
        if not sample.handcrafted:
            continue

        # НАЧАЛО ОТЛАДКИ
        #if 'зовут' not in sample.left or sample.short_phrase.lower() != 'илья':
        #    continue
        # КОНЕЦ ОТЛАДКИ

        if sample.left:
            context = [s.strip() for s in sample.left.split('|')] + [sample.short_phrase]
            expanded_tokens = lemmatizer.lemmatize(tagger.tag(tokenizer.tokenize(sample.expanded_phrase)))
            context_templates = [create_context_template(iline, line_str, expanded_tokens) for iline, line_str in enumerate(context)]
            if any((z is None) for z in context_templates):
                continue

            expanded_template = create_expanded_template(context_templates, expanded_tokens)

            # выкидываем из контекста леммы, так они были нужны только для
            context_templates = tuple(t.generate_packed_template() for t in context_templates)
            k = (context_templates, expanded_template)
            templates2[k] += 1
            packed2samples[k].append(sample)

    print('{} template samples created'.format(len(templates2)))
    # отсортируем шаблоны в порядке убывания частоты
    templates2 = [t for t, freq in templates2.most_common()]
    with open(os.path.join(tmp_dir, 'interpreter_templates2.bin'), 'wb') as f:
        pickle.dump(templates2, f)

    with io.open(os.path.join(tmp_dir, 'interpreter_templates2.txt'), 'w', encoding='utf-8') as wrt:
        for key, samples in packed2samples.items():
            wrt.write('template:\n')
            for i, context in enumerate(key[0]):
                wrt.write('#{}: {}\n'.format(i, context))
            wrt.write('output: {}\n\n'.format(key[1]))

            for sample in samples:
                wrt.write('{}\n{}\n{}\n\n'.format(sample.left.replace(' | ', '\n'), sample.short_phrase, sample.expanded_phrase))

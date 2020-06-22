# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки модели интерпретации ответов в чатботе. Эта модель
восстанавливает полный (подразумеваемый, развернутый) текст ответа по фактическому
краткому ответу собеседника, используя контекст в виде заданного вопроса. Обрабатывается анафора,
гэпинг, эллипсис.

28-03-2020 Добавлен отчет по OOV словам в датасете
30-05-2020 Добавлена загрузка сэмплов из question interpretation датасета в общую кучу
20-06-2020 Добавлена генерация шаблонов интерпретатора (knn1-модель)
"""

import io
import collections
import os
import re
import pickle
import itertools

import rutokenizer
import rupostagger
import ruword2tags
import rulemma


def is_int(s):
    return re.match(r'^\d+$', s)


class Sample(object):
    def __init__(self, question, short_answer, expanded_answer):
        self.question = question
        self.short_answer = short_answer
        self.expanded_answer = expanded_answer


class Sample2:
    def __init__(self, left, short_phrase, expanded_phrase):
        self.left = left
        self.short_phrase = short_phrase
        self.expanded_phrase = expanded_phrase


def create_template_item(token, gren):
    lword = token[0].lower()
    tags = token[1].split('|')
    if tags[0] in ('NOUN', 'VERB', 'ADV', 'ADJ', 'NUM'):
        item = ''
        if tags[0] == 'NOUN':
            item = 'сущ'
            if 'Number=Sing' in tags:
                item += ',ед'

                if 'Gender=Masc' in tags:
                    item += ',муж'
                elif 'Gender=Fem' in tags:
                    item += ',жен'
                elif 'Gender=Neut' in tags:
                    item += ',ср'

            elif 'Number=Plur' in tags:
                item += ',мн'

            if 'Case=Nom' in tags:
                item += ',им'
            elif 'Case=Acc' in tags:
                item += ',вин'
            elif 'Case=Dat' in tags:
                item += ',дат'
            elif 'Case=Ins' in tags:
                item += ',тв'
            elif 'Case=Gen' in tags:
                item += ',род'
            elif 'Case=Prep' in tags:
                item += ',предл'
            elif 'Case=Loc' in tags:
                #item += ',мест'
                item += ',предл'

        elif tags[0] == 'VERB':
            if lword in ('буду', 'будет', 'был', 'была', 'были'):
                return lword

            item = 'гл'

            if 'VerbForm=Conv' in tags:
                item += ',деепр'

            if 'Number=Sing' in tags:
                item += ',ед'
            elif 'Number=Plur' in tags:
                item += ',мн'

            if 'Person=1' in tags:
                item += ',1'
            elif 'Person=2' in tags:
                item += ',2'
            elif 'Person=3' in tags:
                item += ',3'

            if 'Tense=Past' in tags:
                item += ',прош'
            elif 'Tense=Notpast' in tags:
                # будущее для совершенных глаголов будем определять проверять через грам. словарь
                tagsets = gren[token[0]]
                tense = 'наст'
                for tagset in tagsets:
                    if u'ВИД=СОВЕРШ' in tagset:
                        tense = 'буд'
                        break

                item += u',' + tense

            if 'Tense=Past' in tags and 'Number=Sing' in tags:
                if 'Gender=Masc' in tags:
                    item += ',муж'
                elif 'Gender=Fem' in tags:
                    item += ',жен'
                elif 'Gender=Neut' in tags:
                    item += ',ср'

            if 'VerbForm=Inf' in tags:
                item += ',инф'

            if 'Mood=Ind' in tags:
                item += ',изъяв'

        elif tags[0] == 'ADV':
            item = 'нареч'

        elif tags[0] == 'ADJ':
            if lword in 'сам сама само сами мой моя мое мои моего моей моих моим моей моему твой твоя твое твои твоего твоей твоих твоим твоими твоем':
                return lword

            item = 'прил'

            item += ',положит,~кр'

            if 'Number=Sing' in tags:
                item += ',ед'

                if 'Gender=Masc' in tags:
                    item += ',муж'
                elif 'Gender=Fem' in tags:
                    item += ',жен'
                elif 'Gender=Neut' in tags:
                    item += ',ср'

            elif 'Number=Plur' in tags:
                item += ',мн'

            if 'Case=Nom' in tags:
                item += ',им'
            elif 'Case=Acc' in tags:
                item += ',вин'
            elif 'Case=Dat' in tags:
                item += ',дат'
            elif 'Case=Ins' in tags:
                item += ',тв'
            elif 'Case=Prep' in tags:
                item += ',предл'
            elif 'Case=Loc' in tags:
                item += ',предл'
            elif 'Case=Gen' in tags:
                item += ',род'

        elif tags[0] == 'NUM':
            if token[0][0] in '0123456789':
                item = 'num_word'
            else:
                return lword
        else:
            raise NotImplementedError()

        return '['+item.lower()+']'
    else:
        return lword


def create_template(expanded_anser, tokenizer, tagger, gren):
    words = tokenizer.tokenize(expanded_anser)
    tokens = tagger.tag(words)
    template_items = [create_template_item(token, gren) for token in tokens]
    return u' '.join(template_items)


def wx(s):
    return ' '.join(tokenizer.tokenize(s)).lower()


def is_important_token2(pos, lemma):
    if pos in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'NUM'):
        return True

    if lemma in ('да', 'нет', 'не', 'ни', 'ага'):
        return True

    return False


salient_tags = 'Case Gender Number Tense Person Mood'.split()


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

    input_dataset = os.path.join(data_dir, 'interpretation.txt')
    input_dataset2 = os.path.join(data_dir, 'question_interpretation.txt')

    output_dataset = os.path.join(tmp_dir, 'interpreter_templates.tsv')
    output_dataset2 = os.path.join(tmp_dir, 'interpreter_samples.tsv')

    rare_path = os.path.join(tmp_dir, 'most_rare_interpretaions.txt')
    terms_path = os.path.join(tmp_dir, 'interpretation_template_terms.tsv')
    no_expansion_path = os.path.join(tmp_dir, 'interpretation_no_expansion_phrases.txt')

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    samples = []
    samples2 = []
    no_expansion_phrases = []
    with io.open(input_dataset, 'r', encoding='utf-8') as rdr:
        lines = []
        for line in rdr:
            line = line.strip()
            if line:
                lines.append(line)
            else:
                if len(lines) == 2:
                    question = lines[0]
                    if '|' in lines[1]:
                        tx = lines[1].split('|')
                        if len(tx) > 2:
                            print('Data format in line "{}"'.format(lines[1]))
                            exit(0)
                        short_answer = tx[0].strip()
                        expanded_answer = tx[1].strip()
                        if len(expanded_answer) < 2:
                            print('Empty expanded answer for question="{}", short_question="{}"'.format(question, short_answer))

                        sample2 = Sample2(wx(question), wx(short_answer), wx(expanded_answer))
                        samples2.append(sample2)

                        if expanded_answer[-1] in ('.', '?', '!'):
                            expanded_answer = expanded_answer[:-1]

                        sample = Sample(question, short_answer.strip(), expanded_answer)
                        samples.append(sample)

                elif len(lines) == 1:
                    tx = lines[0].split('|')
                    if len(tx) == 2:
                        phrase1 = tx[0].strip()
                        phrase2 = tx[1].strip()
                        if len(phrase2) == 0 or phrase1 == phrase2:
                            no_expansion_phrases.append(u' '.join(tokenizer.tokenize(phrase1)))
                lines = []

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

                    left = ' | '.join(lines[:-1])
                    left = wx(left)
                    short_phrase = wx(short_phrase)
                    expanded_phrase = wx(expanded_phrase)

                    sample2 = Sample2(left, short_phrase, expanded_phrase)
                    samples2.append(sample2)

                lines = []

    print('{} samples, {} no-expansion phrases'.format(len(samples2), len(no_expansion_phrases)))

    print('Writing {} samples to "{}"'.format(len(samples2), output_dataset2))
    output2freq = collections.Counter()
    with io.open(output_dataset2, 'w', encoding='utf-8') as wrt:
        wrt.write('context\toutput\n')
        for sample in samples2:
            context = sample.left + ' | ' + sample.short_phrase
            output = sample.expanded_phrase
            wrt.write('{}\t{}\n'.format(context, output))
            output2freq[output] += 1

    # Сформируем отчет о самых редких интерпретациях, чтобы можно было аугментировать их руками и нарастить частоту.
    with io.open(rare_path, 'w', encoding='utf-8') as wrt:
        rare_outputs = set(output for output, freq in output2freq.items() if freq == 1)
        for sample in sorted(samples2, key=lambda s: (len(s.short_phrase), len(s.left))):
            if sample.expanded_phrase in rare_outputs:
                for context_phrase in sample.left.split('|'):
                    wrt.write('{}\n'.format(context_phrase.strip()))
                wrt.write('{} | {}\n\n\n'.format(sample.short_phrase, sample.expanded_phrase))

    if False:
        d = '/home/inkoziev/polygon/NLPContests/MLBootCamp/2020_26/data'
        train_samples, testval_samples = sklearn.model_selection.train_test_split(samples2, test_size=0.2)
        test_samples, val_samples = sklearn.model_selection.train_test_split(testval_samples, test_size=0.5)

        with io.open(os.path.join(d, 'sent_zh_train.txt'), 'w', encoding='utf-8') as wrt_src, \
             io.open(os.path.join(d, 'sent_ru_train.txt'), 'w', encoding='utf-8') as wrt_trg:
            for sample in train_samples:
                context = sample.left + ' | ' + sample.short_phrase
                wrt_src.write('{}\n'.format(context))
                wrt_trg.write('{}\n'.format(sample.expanded_phrase))

        with io.open(os.path.join(d, 'sent_zh_val.txt'), 'w', encoding='utf-8') as wrt_src, \
             io.open(os.path.join(d, 'sent_ru_val.txt'), 'w', encoding='utf-8') as wrt_trg:
            for sample in val_samples:
                context = sample.left + ' | ' + sample.short_phrase
                wrt_src.write('{}\n'.format(context))
                wrt_trg.write('{}\n'.format(sample.expanded_phrase))

        with io.open(os.path.join(d, 'sent_zh_test.txt'), 'w', encoding='utf-8') as wrt_src, \
             io.open(os.path.join(d, 'sent_ru_test.txt'), 'w', encoding='utf-8') as wrt_trg:
            for sample in test_samples:
                context = sample.left + ' | ' + sample.short_phrase
                wrt_src.write('{}\n'.format(context))
                wrt_trg.write('{}\n'.format(sample.expanded_phrase))

    print('Writing {} samples to "{}"'.format(len(no_expansion_phrases), no_expansion_path))
    with io.open(no_expansion_path, 'w', encoding='utf=8') as wrt:
        for phrase in no_expansion_phrases:
            wrt.write(phrase+'\n')

    tagger = rupostagger.RuPosTagger()
    tagger.load()

    gren = ruword2tags.RuWord2Tags()
    gren.load()

    # НАЧАЛО ОТЛАДКИ
    #words = tokenizer.tokenize('Тебе нравится пить кофе')
    #tags = list(tagger.tag(words))
    # КОНЕЦ ОТЛАДКИ

    lemmatizer = rulemma.Lemmatizer()
    lemmatizer.load()

    all_templates = set()
    template2freq = collections.Counter()
    template2sample = dict()
    all_terms = collections.Counter()

    print('Writing {} samples to "{}"'.format(len(samples), output_dataset))
    with io.open(output_dataset, 'w', encoding='utf=8') as wrt:
        wrt.write('question\tshort_answer\texpanded_answer\ttemplate\n')
        for sample in samples:
            template = create_template(sample.expanded_answer, tokenizer, tagger, gren)
            wrt.write('{}\t{}\t{}\t{}\n'.format(sample.question, sample.short_answer, sample.expanded_answer, template))

            all_templates.add(template)
            template2freq[template] += 1
            template2sample[template] = sample
            all_terms.update(template.split())

    print('all_templates.count={}'.format(len(all_templates)))
    for template, freq in template2freq.most_common(20):
        print('{:<10} {}'.format(freq, template))

    print('all_terms.count={}'.format(len(all_terms)))
    with io.open(terms_path, 'w', encoding='utf-8') as wrt:
        wrt.write('term\tfreq\n')
        for term, freq in all_terms.most_common():
            wrt.write('{}\t{}\n'.format(term, freq))

    # Сформируем отчет о самых редких шаблонах
    #with io.open(rare_path, 'w', encoding='utf-8') as wrt:
    #    for template, freq in reversed(template2freq.most_common()):
    #        if freq < 5:
    #            sample = template2sample[template]
    #            wrt.write('{}\n{}|{}\n{}\nfreq={}\n\n'.format(sample.question, sample.short_answer, sample.expanded_answer, template, freq))

    # Поищем несловарные и нечисловые токены
    vocabulary = set()
    with io.open(os.path.join(data_dir, 'dict/word2lemma.dat'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            fields = line.strip().split('\t')
            if len(fields) == 4:
                word = fields[0].lower().replace(' - ', '-')
                vocabulary.add(word)

    oov_tokens = set()
    with io.open(os.path.join(tmp_dir, 'prepare_interpreter_templates.oov.txt'), 'w', encoding='utf-8') as wrt:
        for sample in samples:
            for phrase in [sample.question, sample.short_answer, sample.expanded_answer]:
                words = tokenizer.tokenize(phrase)
                for word in words:
                    uword = word.lower().replace('ё', 'е')
                    if uword not in vocabulary and not is_int(word) and word not in '. ? ! : - , — – ) ( " \' « » „ “ ; …'.split():
                        if uword not in oov_tokens:
                            wrt.write('Sample with oov-word "{}":\n'.format(word))
                            wrt.write('Question:        {}\n'.format(sample.question))
                            wrt.write('Short answer:    {}\n'.format(sample.short_answer))
                            wrt.write('Expanded answer: {}\n'.format(sample.expanded_answer))
                            wrt.write('\n\n')
                            oov_tokens.add(uword)
                            break

    # Делаем морфологические шаблоны для модели интерпретации knn-1
    print('Building knn1 templates from {} samples...'.format(len(samples2)))
    templates2 = collections.Counter()
    packed2samples = collections.defaultdict(list)
    for sample in samples2:
        # НАЧАЛО ОТЛАДКИ
        #if 'зовут' not in sample.left or sample.short_phrase.lower() != 'илья':
        #    continue
        # КОНЕЦ ОТЛАДКИ

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

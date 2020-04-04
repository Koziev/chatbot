# -*- coding: utf-8 -*-
"""
Подготовка датасета для тренировки модели интерпретации ответов в чатботе. Эта модель
восстанавливает полный (подразумеваемый, развернутый) текст ответа по фактическому
краткому ответу собеседника, используя контекст в виде заданного вопроса.

28-03-2020 Добавлен отчет по OOV словам в датасете
"""

import io
import collections
import os
import re

import rutokenizer
import rupostagger
import ruword2tags


def is_int(s):
    return re.match(r'^\d+$', s)


class Sample(object):
    def __init__(self, question, short_answer, expanded_answer):
        self.question = question
        self.short_answer = short_answer
        self.expanded_answer = expanded_answer


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


if __name__ == '__main__':
    data_dir = '../../data'
    tmp_dir = '../../tmp'

    input_dataset = os.path.join(data_dir, 'interpretation.txt')
    output_dataset = os.path.join(tmp_dir, 'interpreter_templates.tsv')
    rare_path = os.path.join(tmp_dir, 'most_rare_interpretaions.txt')
    terms_path = os.path.join(tmp_dir, 'interpretation_template_terms.tsv')
    no_expansion_path = os.path.join(tmp_dir, 'interpretation_no_expansion_phrases.txt')

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    samples = []
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
                        short_answer, expanded_answer = tx[0], tx[1]
                        expanded_answer = expanded_answer.strip()
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

    print('{} samples, {} no-expansion phrases'.format(len(samples), len(no_expansion_phrases)))

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

    #lemmatizer = rulemma.Lemmatizer()
    #lemmatizer.load()

    all_templates = set()
    template2freq = collections.Counter()
    template2sample = dict()
    all_terms = collections.Counter()

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
    with io.open(rare_path, 'w', encoding='utf-8') as wrt:
        for template, freq in reversed(template2freq.most_common()):
            if freq < 5:
                sample = template2sample[template]
                wrt.write('{}\n{}|{}\n{}\nfreq={}\n\n'.format(sample.question, sample.short_answer, sample.expanded_answer, template, freq))

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



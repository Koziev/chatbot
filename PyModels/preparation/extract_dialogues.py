# -*- coding: utf-8 -*-

import io
import re
import glob


input_file = '../../data/sample_text.txt'
output_file = '../../tmp/dialogues.txt'


class TextReader(object):
    def __init__(self, path):
        self.path = path
        self.unread_line = None
        self.eof_flag = False

    def __enter__(self):
        self.rdr = io.open(self.path, 'r', encoding='utf-8')
        return self

    def __exit__(self, type, value, traceback):
        #Exception handling here
        self.rdr.close()
        self.rdr = None

    def __readline(self):
        retline = None
        if self.unread_line:
            retline = self.unread_line
            self.unread_line = None
        else:
            retline = self.rdr.readline()
            if not retline:
                self.eof_flag = True
            else:
                retline = retline.strip()

        return retline

    def eof(self):
        return self.eof_flag

    def __readphrase(self):
        phrase = u''
        while not self.eof():
            l = self.__readline()
            if l:
                phrase += u' ' + l

                if l[-1] in u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя,:':
                    next_line = self.__readline()
                    self.unread_line = next_line
                    if next_line and next_line[0] in u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
                        pass
                    else:
                        break
                else:
                    break
            else:
                break

        return phrase

    def readphrase(self):
        return self.__readphrase()


def clean_line(line):
    s = line.strip()
    s = s.replace(u' ', u' ').replace(u'  ', u' ').replace(u'--', u'-').replace(u'—', u'-')
    s = s.replace(u'- -', u'-')

    # - Конечно, вижу! - ответил Дрозд. - И не получится, если ты будешь закрывать оба глаза.
    # - Ура! - крикнул пока ещё обыкновенный Домоседов. - Бороться и искать!
    # - О нет, - сказала Фредерика. - Прошлый раз ты меня обманул со своим грибом.
    # - Ваша светлость, я исполнил то, что вы требовали! - торжественно заявил мой алхимик. - Но так как золото добыто волшебным путем
    # - Не шуми, - откликнулся отец: - проснется мать, тогда вот поймешь!
    s = re.sub(u'([!?,]) [—-] ([АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789, ]+?)[.:] [—-] ', u'\\1 ', s)

    # — Ничего, — говорит Валя. — У поросят ведь домики бывают без окон.
    #         ^^^^^^^^^^^^^^^^^^^
    s = re.sub(u', [—-] (.+?). [—-] ([АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯ])', u'. \\2', s)

    # - Пух, - вкрадчиво начал он, потому что не хотел, чтоб Пух подумал, будто бы он сдается, - мне тут в голову пришла одна мысль.
    # - Ну, кажется, всем я угодила, - сказала Весна, - теперь все довольны
    s = re.sub(u'(,) [—-] ([АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789, ]+?), [—-] ', u'\\1 ', s)

    # — Что ты? — спрашивает.
    #          ^^^^^^^^^^^^^^
    # - Черт побери, это же Ленка! - первым узнал Андрей, и включил в комнате свет.
    # - До свиданья, - торопливо сказал Семка.
    # - Вы что, дяденька? - спросил, подбегая, Пашка.
    # - Перси, - автоматически поправил Рон.
    s = re.sub(u'([,?!]) [—-] ([АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789, ]+?)\\.', u'\\1', s)

    # - Ишь что удумали, гады! - растерянно шепнул Пашка через плечо
    s = re.sub(u'([,?!]) [—-] ([АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789, ]+)$', u'\\1', s)

    s = s.replace(u'  ', u' ')

    return s


def is_good_replica(replica):
    return re.match(u'^- ([АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789, ]+)[?!.]$', replica)


def store_dialog(wrt, dialog):
    # Проверим, что в диалоге нет слишком длинных реплик.
    n_stored = 0
    if max(map(len, dialog)) <= 60:
        for dial_len in range(2, len(dialog)+1):
            if is_good_replica(dialogue[dial_len-1]):
                n_stored += 1
                for s in dialogue[:dial_len]:
                    wrt.write(u'{}\n'.format(s))
                wrt.write(u'\n')
    return n_stored


if __name__ == '__main__':
    s2 = clean_line(u'— Ничего, — говорит Валя. — У поросят ведь домики бывают без окон.')
    s3 = clean_line(u'— И я, — говорю, — нет. Я даже себя не вижу.')
    s4 = clean_line(u'— Что ты? — спрашивает.')
    s5 = clean_line(u'- До свиданья, - торопливо сказал Семка.')
    s6 = clean_line(u'- Пух, - вкрадчиво начал он, потому что не хотел, чтоб Пух подумал, будто бы он сдается, - мне тут в голову пришла одна мысль.')
    s7 = clean_line(u'- Перси, - автоматически поправил Рон.')
    s8 = clean_line(u'- Ну, кажется, всем я угодила, - сказала Весна, - теперь все довольны')
    s9 = clean_line(u'- Ваша светлость, я исполнил то, что вы требовали! - торжественно заявил мой алхимик. - Но так как золото добыто волшебным путем')
    s10 = clean_line(u'- Ура! - крикнул пока ещё обыкновенный Домоседов. - Бороться и искать!')
    s11 = clean_line(u'- Ваша светлость, я исполнил то, что вы требовали! - торжественно заявил мой алхимик. - Но так как золото добыто волшебным путем')
    s12 = clean_line(u'- Конечно,  вижу!  -  ответил  Дрозд.  -  И  не  получится, если ты будешь закрывать оба глаза. Закрой только левый, а правый оставь открытым.')
    s13 = clean_line(u'- О нет, - сказала Фредерика. - Прошлый раз ты меня обманул со своим грибом. Он так и не вернулся. И мне попало от мамы.')
    s14 = clean_line(u'- Ишь что удумали, гады! - растерянно шепнул Пашка через плечо')
    s15 = clean_line(u'- Черт  побери,  это  же  Ленка!  -  первым  узнал  Андрей, и включил в комнате свет.')

    with io.open(output_file, 'w', encoding='utf-8') as wrt:
        total_nb_dialogues = 0
        for filename in glob.iglob(u'/media/inkoziev/corpora/Corpus/Raw/ru/Texts/**/*.txt', recursive=True):
        #for filename in ['../../data/sample_text.txt']:
            print(u'Processing {}...'.format(filename))
            nb_extracted = 0
            dialogue = []
            with TextReader(filename) as rdr:
                prev_line = ''
                while not rdr.eof():
                    line = rdr.readphrase()
                    if line:
                        line = line.strip()
                        if line[0] in u'—-':
                            if prev_line and prev_line[0] not in u'—-':
                                if len(dialogue) >= 2:
                                    # Не будем сохранять диалоги из 1 реплики.
                                    n = store_dialog(wrt, dialogue)
                                    nb_extracted += n
                                    total_nb_dialogues += n
                                    dialogue = []

                            s = clean_line(line)
                            if len(set(s)) > 1:  # попадаются строки, состоящие из одних "-"
                                #if s.startswith(u'- Черт побери, это же Ленка! - первым узнал Андрей, и включил в комнате свет.'):
                                #    print('DEBUG@143')
                                #    print(u'line={}'.format(line))
                                #    print(u's={}'.format(s))
                                #    exit(0)
                                dialogue.append(s)

                        prev_line = line

            print(u'{} dialogues extracted from {}'.format(nb_extracted, filename))

    print('All done, {} dialogues extracted and stored'.format(total_nb_dialogues))


# -*- coding: utf-8 -*-
'''
'''

from __future__ import division  # for python2 compatibility
from __future__ import print_function

import io


corpus = r'/media/inkoziev/corpora/Corpus/word2vector/ru/w2v.ru.corpus.txt'
max_len = 80

bad_words = set(u'ее его их нас вас она он вы мы мной те эти тот этот та эта то это той тем теми этой этим этими тех этих'.split())

hits = 0
with io.open('../../tmp/tmp_facts.txt', 'w', encoding='utf-8') as wrt,\
     io.open(corpus, 'r', encoding='utf-8') as f:
    for line in f:
        s = line.strip()
        if s[-1] == '.':
            if len(s) <= max_len:
                words = s.split()
                if u'я' in words and u'люблю' in words and u'задачи' in words:
                    #if u'в' not in words:
                    #if u'выключился' in s:
                    if len(set(words).intersection(bad_words)) == 0:
                        #if s == u'немецкий фольксваген не состоит на 1 % из немецких материалов .':
                        #    pass

                        wrt.write(u'{}\n'.format(s))
                        hits += 1
                        if hits >= 10000:
                            break

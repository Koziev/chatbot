# -*- coding: utf-8 -*-
"""
Сегментатор - разбивает текст на предложения.
"""

from __future__ import print_function
from __future__ import division  # for python2 compatability
import ruchatbot.utils.abbrev
import ruchatbot.utils.textnormalizer


class Segmenter(object):
    def __init__(self):
        #self.regex = re.compile(r'[\\.\\?\\!;]')
        pass

    def is_name_delim(self, c):
        return c in u' \t,;'

    def is_cyr(self, c):
        # cyrillic 0x0400...0x04FF
        return 0x0400 <= ord(c) <= 0x04ff

    def split(self, text0):
        """
        :param text0: text to split
        :return: list of sentences
        """
        text = utils.abbrev.normalize_abbrev(text0)
        text = utils.textnormalizer.preprocess_line(text)
        res = []
        break_pos = -1
        full_len = len(text)
        while break_pos < full_len:
            next_break = full_len
            for break_char in u'.?？!;':

                start_pos = break_pos + 1
                while start_pos != -1 and start_pos < full_len:
                    p = text.find(break_char, start_pos)
                    if p == -1:
                        break

                    if p < next_break:
                        if break_char == u'.' and p > 1 and text[p - 1].isupper() and self.is_name_delim(text[p - 2]):
                            #  А. С. Пушкин
                            #   ^
                            start_pos = p+1
                            continue

                        if break_char in u',.' and p > 0 and p < full_len - 1 and text[p - 1].isdigit() and text[p + 1].isdigit():
                            # 3.141
                            start_pos = p+1
                            continue

                        if break_char == u'.' and p > 1 and text[p - 2] == u' ' and self.is_cyr(text[p - 1]):
                            # приехал из с. Зимнее
                            #             ^
                            start_pos = p+1
                            continue

                        next_break = p
                        break
                    else:
                        break

            sent = text[break_pos + 1: next_break + 1].strip()
            if len(sent) > 1:
                res.append(sent)
            break_pos = next_break

        return res

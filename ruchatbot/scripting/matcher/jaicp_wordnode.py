""" Узел AST для одного слова, или леммы, или стема """

import re
import string
import logging

from .jaicp_basenode import JAICP_BaseNode
from .matching import Matching


def ngrams(s, n):
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len=3):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)


class JAICP_WordNode(JAICP_BaseNode):
    TYPE_LEXEM = 0
    TYPE_STAR_STEM = 1
    TYPE_STEM_STAR = 2
    TYPE_STAR_STEM_STAR = 3
    TYPE_LEMMA = 4
    TYPE_ONEWORD = 6
    TYPE_NONEMPTYGARBAGE = 7

    def __init__(self, src_str):
        super(JAICP_WordNode, self).__init__(src_str)
        self.type = -1
        self.lemma = None

        if src_str == '$oneWord':
            self.type = JAICP_WordNode.TYPE_ONEWORD
        elif src_str == '$nonEmptyGarbage':
            self.type = JAICP_WordNode.TYPE_NONEMPTYGARBAGE
        elif src_str.startswith('~'):
            self.type = JAICP_WordNode.TYPE_LEMMA
            self.lemma = src_str[1:].strip()
        elif src_str.startswith('*') and src_str.endswith('*'):
            self.type = JAICP_WordNode.TYPE_STAR_STEM_STAR
        elif src_str.startswith('*'):
            self.type = JAICP_WordNode.TYPE_STAR_STEM
        elif src_str.endswith('*'):
            self.type = JAICP_WordNode.TYPE_STEM_STAR
        #elif src_str.startswith('$'):
        #    self.type = JAICP_WordNode.TYPE_NAMEDLIST
        #    self.named_list_items = []  # TODO!
        else:
            self.type = JAICP_WordNode.TYPE_LEXEM

    def __repr__(self):
        return self.src_str + self.get_next_node_repr()

    def match(self, words, start_index, cache):
        if start_index >= len(words):
            # Выход за правую границу предложения
            return []

        word = words.get_word(start_index)
        uword = word.lower()

        success = False
        score = 0.0
        if self.type == JAICP_WordNode.TYPE_LEXEM:
            try:
                success = re.match(self.src_str, word, flags=re.IGNORECASE) is not None
                score = 1.0
            except re.error:
                logging.error('Error when matching regexp "%s" with string "%s"', self.src_str, word)
                raise
        elif self.type == JAICP_WordNode.TYPE_STAR_STEM:
            stem = self.src_str[1:].lower()
            success = uword.endswith(stem)
            score = jaccard(uword, stem)
        elif self.type == JAICP_WordNode.TYPE_STEM_STAR:
            stem = self.src_str[:-1].lower()
            score = jaccard(uword, stem)
            success = uword.startswith(stem)
        elif self.type ==  JAICP_WordNode.TYPE_STAR_STEM_STAR:
            stem = self.src_str[1:-1].lower()
            score = jaccard(uword, stem)
            success = stem in uword
        elif self.type == JAICP_WordNode.TYPE_LEMMA:
            success = words.get_lemma(start_index).lower() == self.lemma
            score = 0.9
        elif self.type == JAICP_WordNode.TYPE_ONEWORD:
            success = True
            score = 0.1
        elif self.type == JAICP_WordNode.TYPE_NONEMPTYGARBAGE:
            success = word[0] not in string.punctuation
            score = 0.1
        else:
            raise RuntimeError()

        if success:
            if self.next_node is None:
                mm = Matching()
                mm.from_itoken = start_index
                mm.nb_words = 1
                mm.hits = score
                return [mm]
            else:
                mx2 = self.next_node.match(words, start_index + 1, cache)
                res_mx = []
                if mx2:
                    for m2 in mx2:
                        mm = Matching()
                        mm.from_itoken = start_index
                        mm.nb_words = m2.nb_words + 1
                        mm.hits = score
                        mm.inner_matchings.append(m2)
                        res_mx.append(mm)
                return self.limit(res_mx)
        else:
            return []

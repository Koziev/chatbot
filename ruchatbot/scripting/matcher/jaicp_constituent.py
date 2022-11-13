""" Узел AST для конструкции ⟦ ... ⟧ - составляющая. """

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode


class JAICP_Constituent(JAICP_BaseNode):
    def __init__(self, src_str):
        super(JAICP_Constituent, self).__init__(src_str)
        self.is_lemma = None
        self.head_lemma = None
        self.head_form = None

        if src_str[0] == '~':
            self.is_lemma = True
            self.head_lemma = src_str[1:].strip()
        else:
            self.is_lemma = False
            self.head_form = src_str.strip()

    def __repr__(self):
        return '⟦' + self.src_str + '⟧' + self.get_next_node_repr()

    def match_head(self, words, start_index):
        if self.is_lemma:
            return words.get_lemma(start_index) == self.head_lemma
        else:
            return words.get_word(start_index) == self.head_form

    def match0(self, words, start_index):
        if self.match_head(words, start_index):
            # Пропускаем всю составляющую, вершиной которой служит текущий токен
            tx = words.extract_constituent(words.get_token(start_index))
            tx = sorted(tx, key=lambda z: int(z.id))
            last_id = int(tx[-1].id) - 1
            mm = Matching()
            mm.from_itoken = start_index
            mm.nb_words = last_id - start_index + 1
            mm.hits = 1.0
            return mm
        else:
            udp_token = words.get_token(start_index)
            head_index = int(udp_token.head) - 1
            if head_index != 0:
                if self.match_head(words, head_index):
                    tx = words.extract_constituent(words.get_token(head_index))
                    tx = sorted(tx, key=lambda z: int(z.id))
                    last_id = int(tx[-1].id) - 1
                    mm = Matching()
                    mm.from_itoken = start_index
                    mm.nb_words = last_id - start_index + 1
                    mm.hits = 1.0
                    return mm
        return None

    def match(self, words, start_index, cache):
        m1 = self.match0(words, start_index)
        if m1:
            if self.next_node is None:
                return [m1]
            else:
                res_mx = []
                mx2 = self.next_node.match(words, start_index + m1.nb_words, cache)
                for m2 in mx2:
                    mm = Matching()
                    mm.from_itoken = start_index
                    mm.nb_words = m1.nb_words + m2.nb_words
                    mm.inner_matchings.append(m1)
                    mm.inner_matchings.append(m2)
                    res_mx.append(mm)

                return self.limit(res_mx)

        return []


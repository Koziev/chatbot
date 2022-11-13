""" Узел AST паттерна для оператора $regex<...> """

import re

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode


class JAICP_RegexNode(JAICP_BaseNode):
    def __init__(self, src_str):
        super(JAICP_RegexNode, self).__init__(src_str)
        try:
            self.rx = re.compile(src_str, flags=re.IGNORECASE)
        except:
            print('Error when compiling regexp: {}'.format(src_str))
            raise

    def __repr__(self):
        return '$regex<' + self.src_str + '>' + self.get_next_node_repr()

    def match(self, words, start_index, cache):
        if start_index >= len(words):
            return []

        word = words.get_word(start_index)
        mx = self.rx.match(word)
        if mx is not None:
            if self.next_node is None:
                mm = Matching()
                mm.from_itoken = start_index
                mm.hits = 1.0
                mm.nb_words = 1
                return [mm]
            else:
                mx2 = self.next_node.match(words, start_index + 1, cache)
                res_mx = []
                if mx2:
                    for m2 in mx2:
                        mm = Matching()
                        mm.from_itoken = start_index
                        mm.nb_words = m2.nb_words + 1
                        mm.hits = 1.0
                        mm.inner_matchings.append(m2)
                        res_mx.append(mm)
                return self.limit(res_mx)
        else:
            return []


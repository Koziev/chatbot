""" Узел AST для оператора * """

import re

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode


class JAICP_StarNode(JAICP_BaseNode):
    def __init__(self, src_str):
        super(JAICP_StarNode, self).__init__(src_str)
        self.min_count = 0
        self.max_count = 1000000

        if '{' in src_str:
            m = re.match(r'\*\{(\d+)\}', src_str)
            if m:
                self.min_count = int(m.group(1))
                self.max_count = self.min_count
            else:
                m = re.match(r'\*\{(\d+),(\d+)\}', src_str)
                if m:
                    self.min_count = int(m.group(1))
                    self.max_count = int(m.group(2))
                else:
                    m = re.match(r'\*\{(\d+),\}', src_str)
                    if m:
                        self.min_count = int(m.group(1))
                        self.max_count = 1000000
                    else:
                        m = re.match(r'\*\{,(\d+)\}', src_str)
                        if m:
                            self.min_count = 0
                            self.max_count = int(m.group(2))

        self.next_node = None

    def __repr__(self):
        return self.src_str + self.get_next_node_repr()

    def is_star(self):
        return True

    def match(self, words, start_index, cache):
        max_count = min(len(words) - start_index - 1, self.max_count)
        res_mx = []
        if self.next_node is None:
            # Финальный * ???
            mm = Matching()
            mm.from_itoken = start_index
            mm.nb_words = len(words) - start_index
            mm.penalty = 0
            mm.hits = 0.01 * mm.nb_words
            res_mx.append(mm)
        else:
            for n_skip in range(self.min_count, max_count+1):
                mx2 = self.next_node.match(words, start_index + n_skip, cache)
                for m2 in mx2:
                    mm = Matching()
                    mm.from_itoken = start_index
                    mm.nb_words = n_skip + m2.nb_words
                    mm.penalty = 0.1 * n_skip
                    mm.inner_matchings.append(m2)
                    res_mx.append(mm)

        return self.limit(res_mx)


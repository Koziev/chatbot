""" Узел AST для вызова именованного паттерна """

import logging

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode


class JAICP_NamedPatternWithLabel(JAICP_BaseNode):
    def __init__(self, pattern_name, label):
        if label is None:
            src_str = pattern_name
        else:
            src_str = pattern_name + '::' + label

        super(JAICP_NamedPatternWithLabel, self).__init__(src_str)
        self.pattern_name = pattern_name
        self.label = label
        self.node = None
        self.bound_pattern = None

    def bind_named_patterns(self, named_patterns):
        if self.node is None:
            if self.pattern_name not in named_patterns:
                logging.error('Missing named pattern "%s"', self.pattern_name)
                raise KeyError(self.pattern_name)
            else:
                self.bound_pattern = named_patterns[self.pattern_name]
                self.node = named_patterns[self.pattern_name].start_node

            super(JAICP_NamedPatternWithLabel, self).bind_named_patterns(named_patterns)

    def list_all_nodes(self, nodes):
        super(JAICP_NamedPatternWithLabel, self).list_all_nodes(nodes)
        if self.node is not None:
            self.node.list_all_nodes(nodes)

    def match(self, words, start_index, cache):
        if self.node is None:
            return []

        mx0 = cache.get(start_index, self.pattern_name)
        if mx0 is not None:
            if self.next_node is None:
                return mx0
            else:
                res_mx = []
                for m1 in mx0:
                    mx2 = self.next_node.match(words, m1.from_itoken+m1.nb_words, cache)
                    if mx2:
                        for m2 in mx2:
                            mm = Matching()
                            mm.from_itoken = m1.from_itoken
                            mm.nb_words = m1.nb_words + m2.nb_words
                            mm.inner_matchings = [m1, m2]
                            mm.label = None
                            res_mx.append(mm)

                return self.limit(res_mx)

        self.bound_pattern.check_count += 1

        mx = self.node.match(words, start_index, cache)
        assert(mx is not None)
        cache.store(start_index, self.pattern_name, mx)

        if mx:
            if self.label is None:
                mx1 = mx
            else:
                mx1 = []
                for m1 in mx:
                    mm = Matching()
                    mm.from_itoken = m1.from_itoken
                    mm.nb_words = m1.nb_words
                    mm.inner_matchings = [m1]
                    mm.label = self.label
                    mx1.append(mm)

            if self.next_node is None:
                return mx1
            else:
                res_mx = []
                for m1 in mx1:
                    mx2 = self.next_node.match(words, m1.from_itoken+m1.nb_words, cache)
                    if mx2:
                        for m2 in mx2:
                            mm = Matching()
                            mm.from_itoken = m1.from_itoken
                            mm.nb_words = m1.nb_words + m2.nb_words
                            mm.inner_matchings = [m1, m2]
                            mm.label = None
                            res_mx.append(mm)

                return self.limit(res_mx)
        else:
            return []

""" Узел AST паттерна - опциональный элемент [...] """

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode
from .matching_cache import MatchingCache
from .parsing_result import ParsingResult


class JAICP_OptionalNode(JAICP_BaseNode):
    def __init__(self, src_str: str):
        super(JAICP_OptionalNode, self).__init__(src_str)
        self.node = None

    def __repr__(self):
        return '[' + self.src_str + ']' + self.get_next_node_repr()

    def bind_named_patterns(self, named_patterns):
        if self.node is not None:
            self.node.bind_named_patterns(named_patterns)
        super(JAICP_OptionalNode, self).bind_named_patterns(named_patterns)

    def list_all_nodes(self, nodes):
        super(JAICP_OptionalNode, self).list_all_nodes(nodes)
        self.node.list_all_nodes(nodes)

    def optimize(self):
        super(JAICP_OptionalNode, self).optimize()
        self.node = self.node.optimize()
        assert(self.node is not None)
        return self

    def match(self, words: ParsingResult, start_index: int, cache: MatchingCache) -> [Matching]:
        # mx0 = cache.get_by_id(start_index, id(self))
        # if mx0 is not None:
        #     if self.next_node is None:
        #         return mx0
        #     else:
        #         res_mx = []
        #         for m1 in mx0:
        #             mx2 = self.next_node.match(words, m1.from_itoken+m1.nb_words, cache)
        #             if mx2:
        #                 for m2 in mx2:
        #                     mm = Matching()
        #                     mm.from_itoken = m1.from_itoken
        #                     mm.nb_words = m1.nb_words + m2.nb_words
        #                     mm.inner_matchings = [m1, m2]
        #                     mm.label = None
        #                     res_mx.append(mm)
        #
        #         return self.limit(res_mx)

        res_mx = []
        mx = self.node.match(words, start_index, cache)

        # Случай отсутствия сопоставления - приемлемый вариант для optional
        amx = list(mx)

        # Добавляем вариант БЕЗ сопоставления, даже если сопоставление найдено.
        # Это нужно, чтобы верно работали операторы типа [$oneWord]
        mm = Matching()
        mm.from_itoken = start_index
        mm.nb_words = 0
        amx.append(mm)

        if self.next_node is None:
            return amx
        else:
            for m in amx:
                start_index2 = start_index + m.nb_words
                mx2 = self.next_node.match(words, start_index2, cache)
                if mx2:
                    for m2 in mx2:
                        mm = Matching()
                        mm.from_itoken = start_index
                        mm.nb_words = m2.nb_words + m.nb_words
                        mm.inner_matchings.append(m)
                        mm.inner_matchings.append(m2)
                        res_mx.append(mm)

        res_mx = self.limit(res_mx)
        #cache.store_by_id(start_index, id(self), res_mx)
        return res_mx

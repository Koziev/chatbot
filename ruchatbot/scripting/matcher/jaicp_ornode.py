""" Узел AST паттерна для альтернативы (.../.../...) """

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode
from .jaicp_wordnode import JAICP_WordNode
from .matching_cache import MatchingCache
from .parsing_result import ParsingResult


class JAICP_OrNode(JAICP_BaseNode):
    def __init__(self, src_str: str, alternative_nodes):
        super(JAICP_OrNode, self).__init__(src_str)
        self.nodes = alternative_nodes

    def __repr__(self):
        return '(' + self.src_str + ')' + self.get_next_node_repr()

    def bind_named_patterns(self, named_patterns):
        for n, _ in self.nodes:
            n.bind_named_patterns(named_patterns)
        super(JAICP_OrNode, self).bind_named_patterns(named_patterns)

    def list_all_nodes(self, nodes):
        super(JAICP_OrNode, self).list_all_nodes(nodes)
        for n, label in self.nodes:
            n.list_all_nodes(nodes)

    @staticmethod
    def is_lemma_node(n):
        return isinstance(n, JAICP_WordNode) and n.type == JAICP_WordNode.TYPE_LEMMA

    def optimize(self):
        super(JAICP_OrNode, self).optimize()

        if all(JAICP_OrNode.is_lemma_node(n) for n, _ in self.nodes):
            lemmas = [n.lemma for n, _ in self.nodes]
            n = JAICP_OrLemmas(self, lemmas)
            print('DEBUG@36 len(lemmas)={}'.format(len(lemmas)))
            return n
        else:
            new_nodes = []
            for n, label in self.nodes:
                new_nodes.append((n.optimize(), label))
            self.nodes = new_nodes
            return self

    def match(self, words: ParsingResult, start_index: int, cache: MatchingCache) -> [Matching]:
        mx0 = cache.get_by_id(start_index, id(self))
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

        res_mx = []
        for item, value in self.nodes:
            mx1 = item.match(words, start_index, cache)
            if mx1:
                for m1 in mx1:
                    if self.next_node is None:
                        res_mx.append(m1)
                    else:
                        mx2 = self.next_node.match(words, start_index + m1.nb_words, cache)
                        if mx2:
                            for m2 in mx2:
                                mm = Matching()
                                mm.from_itoken = start_index
                                mm.nb_words = m1.nb_words + m2.nb_words
                                mm.inner_matchings.append(m1)
                                mm.inner_matchings.append(m2)
                                res_mx.append(mm)
                break

        res_mx = self.limit(res_mx)
        cache.store_by_id(start_index, id(self), res_mx)
        return res_mx


class JAICP_OrLemmas(JAICP_BaseNode):
    def __init__(self, src_ornode, lemmas):
        super(JAICP_OrLemmas, self).__init__(src_ornode.src_str)
        self.lemmas = list(lemmas)
        self.next_node = src_ornode.next_node

    def __repr__(self):
        return '(' + self.src_str + ')' + self.get_next_node_repr()

    def bind_named_patterns(self, named_patterns):
        super(JAICP_OrLemmas, self).bind_named_patterns(named_patterns)

    def list_all_nodes(self, nodes):
        super(JAICP_OrLemmas, self).list_all_nodes(nodes)

    def match(self, words: ParsingResult, start_index: int, cache: MatchingCache) -> [Matching]:
        res_mx = []
        if start_index >= len(words):
            return res_mx

        mx0 = cache.get_by_id(start_index, id(self))
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

        for lemma in self.lemmas:
            if words.get_lemma(start_index).lower() == lemma:
                m1 = Matching()
                m1.from_itoken = start_index
                m1.nb_words = 1
                m1.hits = 0.9

                if self.next_node is None:
                    res_mx.append(m1)
                else:
                    mx2 = self.next_node.match(words, start_index + m1.nb_words, cache)
                    if mx2:
                        for m2 in mx2:
                            mm = Matching()
                            mm.from_itoken = start_index
                            mm.nb_words = m1.nb_words + m2.nb_words
                            mm.inner_matchings.append(m1)
                            mm.inner_matchings.append(m2)
                            res_mx.append(mm)
                break

        res_mx = self.limit(res_mx)
        cache.store_by_id(start_index, id(self), res_mx)
        return res_mx

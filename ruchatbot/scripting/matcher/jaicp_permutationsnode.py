""" Узел AST паттерна для оператора перестановок { ... } """

import copy
import itertools

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode
from .matching_cache import MatchingCache
from .parsing_result import ParsingResult


class JAICP_PermutationsNode(JAICP_BaseNode):
    def __init__(self, src_str):
        super(JAICP_PermutationsNode, self).__init__(src_str)
        self.paths = []

    @staticmethod
    def build(src_str, nodes):
        r = JAICP_PermutationsNode(src_str)

        for items2 in itertools.permutations(nodes):
            items3 = [copy.deepcopy(x) for x in items2]

            # Ищем * на правом конце
            n_stars_tail = 0
            for n in items3[::-1]:
                if n.is_star():
                    n_stars_tail += 1
                else:
                    break

            # Ищем * в начале
            n_stars_head = 0
            for n in items3:
                if n.is_star():
                    n_stars_head += 1
                else:
                    break

            #if n_stars_head > 0:
            #    items3 = items3[n_stars_head:]

            if n_stars_tail > 0:
                items3 = items3[:-n_stars_tail]
                is_open_ended = True
            else:
                is_open_ended = False

            # ПОКА ПОПРОБУЕМ ИСКЛЮЧИТЬ ВАРИАНТЫ С * В НАЧАЛЕ И В КОНЦЕ
            if True:  #n_stars_head == 0 and n_stars_tail == 0:
                # Исключаем два и более оператора * подряд
                items4 = []
                for n1, n2 in zip(items3, items3[1:]):
                    if n1.is_star() and n2.is_star():
                        continue
                    else:
                        items4.append(n1)

                items4.append(items3[-1])

                # делаем связанный список
                for x1, x2 in zip(items4, items4[1:]):
                    x1.next_node = x2

                r.paths.append((items4[0], is_open_ended))

        return r

    def __repr__(self):
        return '{' + self.src_str + '}' + self.get_next_node_repr()

    def bind_named_patterns(self, named_patterns):
        for n, _ in self.paths:
            n.bind_named_patterns(named_patterns)
        super(JAICP_PermutationsNode, self).bind_named_patterns(named_patterns)

    def list_all_nodes(self, nodes):
        super(JAICP_PermutationsNode, self).list_all_nodes(nodes)
        for n, _ in self.paths:
            n.list_all_nodes(nodes)

    def optimize(self):
        super(JAICP_PermutationsNode, self).optimize()
        new_paths = []
        for n, f in self.paths:
            new_paths.append((n.optimize(), f))
        self.paths = new_paths
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
        for n, is_open_ended in self.paths:
            mx = n.match(words, start_index, cache)
            if is_open_ended:
                for m1 in mx:
                    max_tail_len = len(words) - start_index - m1.nb_words
                    for tail_len in range(0, max_tail_len+1):
                        if self.next_node is None:
                            mm12 = Matching()
                            mm12.from_itoken = start_index
                            mm12.nb_words = m1.nb_words + tail_len
                            mm12.inner_matchings.append(m1)
                            res_mx.append(mm12)
                        else:
                            mx2 = self.next_node.match(words, start_index + m1.nb_words + tail_len, cache)
                            if mx2:
                                for m2 in mx2:
                                    mm = Matching()
                                    mm.from_itoken = start_index
                                    mm.nb_words = m1.nb_words + tail_len + m2.nb_words
                                    mm.inner_matchings.append(m1)
                                    mm.inner_matchings.append(m2)
                                    res_mx.append(mm)

            else:
                if self.next_node is None:
                    res_mx.extend(mx)
                else:
                    for m1 in mx:
                        mx2 = self.next_node.match(words, start_index + m1.nb_words, cache)
                        if mx2:
                            for m2 in mx2:
                                mm = Matching()
                                mm.from_itoken = start_index
                                mm.nb_words = m1.nb_words + m2.nb_words
                                mm.inner_matchings.append(m1)
                                mm.inner_matchings.append(m2)
                                res_mx.append(mm)

        res_mx = self.limit(res_mx)
        cache.store_by_id(start_index, id(self), res_mx)
        return res_mx


""" Узел AST паттерна для оператора отрицания ^ """

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode


class JAICP_NegationNode(JAICP_BaseNode):
    def __init__(self, src_str, arg_node):
        super(JAICP_NegationNode, self).__init__(src_str)
        self.arg_node = arg_node

    def __repr__(self):
        return '^' + str(self.arg_node) + self.get_next_node_repr()

    def bind_named_patterns(self, named_patterns):
        if self.arg_node is not None:
            self.arg_node.bind_named_patterns(named_patterns)
        super(JAICP_NegationNode, self).bind_named_patterns(named_patterns)

    def list_all_nodes(self, nodes):
        super(JAICP_NegationNode, self).list_all_nodes(nodes)
        self.arg_node.list_all_nodes(nodes)

    def optimize(self):
        super(JAICP_NegationNode, self).optimize()
        self.arg_node = self.arg_node.optimize()
        return self

    def match(self, words, start_index, cache):
        mx = self.arg_node.match(words, start_index, cache)
        if mx:
            return []
        else:
            if self.next_node is None:
                mm = Matching()
                mm.from_itoken = start_index
                mm.nb_words = 0
                return [mm]
            else:
                mx2 = self.next_node.match(words, start_index, cache)
                return mx2

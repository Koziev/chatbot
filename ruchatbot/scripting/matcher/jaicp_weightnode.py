""" Узел AST для оператора $weight<...> """


from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode

class JAICP_WeightNode(JAICP_BaseNode):
    def __init__(self, w_str):
        src_str = '$weight<' + w_str + '>'
        super(JAICP_WeightNode, self).__init__(src_str)
        self.a = 1.0
        self.b = 0.0

        if '+' in w_str:
            str_a, str_b = w_str.split('+')
            str_a = str_a.strip()
            if str_a:
                self.a = float(str_a)
            self.b = float(str_b.strip())
        elif '-' in w_str:
            str_a, str_b = w_str.split('-')
            str_a = str_a.strip()
            if str_a:
                self.a = float(str_a.strip())
            self.b = -float(str_b.strip())
        else:
            self.a = float(w_str)
            self.b = 0.0

    def match(self, words, start_index, cache):
        if self.next_node is None:
            mm = Matching()
            mm.from_itoken = start_index
            mm.nb_words = 0
            return [mm]
        else:
            mx = self.next_node.match(words, start_index, cache)
            return mx

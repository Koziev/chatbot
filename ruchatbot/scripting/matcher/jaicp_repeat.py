""" Узел AST паттерна для оператора $repeat<...> """

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode


class JAICP_Repeat(JAICP_BaseNode):
    def __init__(self, src_str, pattern_name, pattern_body):
        super(JAICP_Repeat, self).__init__(src_str)
        self.pattern_name = pattern_name
        self.pattern_body = pattern_body

    def list_all_nodes(self, nodes):
        super(JAICP_Repeat, self).list_all_nodes(nodes)
        self.pattern_body.list_all_nodes(nodes)

    def bind_named_patterns(self, named_patterns):
        self.pattern_body.bind_named_patterns(named_patterns)
        super(JAICP_Repeat, self).bind_named_patterns(named_patterns)

    @staticmethod
    def parse(tokens, named_patterns):
        tokens.read_it('<')
        pattern_name = tokens.read()
        tokens.read_it('>')
        assert(pattern_name.startswith('$'))
        assert(pattern_name in named_patterns)
        return JAICP_Repeat('$repeat<'+pattern_name+'>', pattern_name, named_patterns[pattern_name].start_node)

    def match(self, words, start_index, cache):
        cur_index = start_index
        seq_matchings = []
        for _ in range(100):
            mx = self.pattern_body.match(words, cur_index, cache)
            if mx:
                mx = sorted(mx, key=lambda z: -z.score())
                m = mx[0]
                seq_matchings.append(m)
                cur_index += m.nb_words
            else:
                if self.next_node:
                    mx2 = self.next_node.match(words, cur_index, cache)
                    res_mx2 = []
                    for m2 in mx2:
                        mm = Matching()
                        mm.from_itoken = start_index
                        mm.nb_words = cur_index - start_index + m2.nb_words
                        mm.hits = 0.0
                        mm.inner_matchings = seq_matchings + [m2]
                        res_mx2.append(mm)

                    return self.limit(res_mx2)
                else:
                    mm = Matching()
                    mm.from_itoken = start_index
                    mm.nb_words = cur_index - start_index
                    mm.hits = 0.0
                    mm.inner_matchings = seq_matchings
                    return [mm]

        raise OverflowError()


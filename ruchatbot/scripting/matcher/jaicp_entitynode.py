""" Узел AST для узла $entity<...> """

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode


class JAICP_EntityNode(JAICP_BaseNode):
    def __init__(self, ref_name):
        src_str = '$entity<'+ref_name+'>'
        super(JAICP_EntityNode, self).__init__(src_str)
        self.ref_name = ref_name
        self.ref = None

    def __repr__(self):
        return '$entity<' + self.ref_name + '>' + self.get_next_node_repr()

    def bind_entities(self, entities):
        self.ref = entities[self.ref_name]

    def match(self, words, start_index, cache):
        max_len = min(self.ref.max_len, len(words) - start_index)
        for n in range(max_len, self.ref.min_len-1, -1):
            probe = ' '.join(words.get_lemmas(start_index, n))
            entity_item = self.ref.find_item(probe)
            if entity_item is not None:
                mm = Matching()
                mm.entity_item = entity_item
                mm.matcher = self
                mm.from_itoken = start_index
                mm.nb_words = n
                mm.hits = n

                if self.next_node is None:
                    return [mm]
                else:
                    mx2 = self.next_node.match(words, start_index + n, cache)
                    res_mx = []
                    if mx2:
                        for m2 in mx2:
                            mm2 = Matching()
                            mm2.from_itoken = start_index
                            mm2.nb_words = n + m2.nb_words
                            mm2.inner_matchings = [mm, m2]
                            res_mx.append(mm2)

                    return self.limit(res_mx)

        return []

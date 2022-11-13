"""Базовый класс узла для AST паттернов"""

from .matching_cache import MatchingCache
from .parsing_result import ParsingResult
from .matching import Matching


class JAICP_BaseNode:
    def __init__(self, src_str: str):
        self.src_str = src_str
        self.next_node = None

    def get_next_node_repr(self) -> str:
        if self.next_node:
            return ' ' + str(self.next_node)
        else:
            return ''

    def __repr__(self):
        return self.src_str + self.get_next_node_repr()

    # @virtual
    def is_star(self) -> bool:
        """Будет перегружен для узла '*' """
        return False

    def list_all_nodes(self, nodes):
        nodes.append(self)
        if self.next_node is not None:
            self.next_node.list_all_nodes(nodes)

    def bind_named_patterns(self, named_patterns) -> None:
        """
        В ходе подготовки паттерна надо привязать именованные паттерны к узлам AST.
        Тут мы спускаемся вниз по AST, позволяя всем дочерним элементам тоже сделать привязку.
        """
        if self.next_node is not None:
            self.next_node.bind_named_patterns(named_patterns)

    def bind_entities(self, entities) -> None:
        pass

    def optimize(self):
        if self.next_node is not None:
            self.next_node = self.next_node.optimize()
            assert(self.next_node is not None)

        return self

    def limit(self, matchings: [Matching]) -> [Matching]:
        """ В ходе сопоставления будем делать beam search, и урезать набор текущих
         сопоставлений на каждом шаге. """
        if len(matchings) > 5:
            return sorted(matchings, key=lambda z: -z.score())[:5]
        else:
            return matchings

    def match(self, words: ParsingResult, start_index: int, cache: MatchingCache) -> [Matching]:
        raise NotImplementedError()

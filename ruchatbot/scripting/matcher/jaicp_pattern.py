"""
Контейнер для сущности "паттерн", включающей в себя как именованные паттерны, так и глобальные
паттерны для стейтов.
"""

from .jaicp_tokenizer import JAICP_Tokenizer
from .jaicp_ornode import JAICP_OrNode
from .jaicp_permutationsnode import JAICP_PermutationsNode
from .jaicp_optionalnode import JAICP_OptionalNode
from .jaicp_starnode import JAICP_StarNode
from .jaicp_regexnode import JAICP_RegexNode
from .jaicp_entitynode import JAICP_EntityNode
from .jaicp_morphnode import JAICP_MorphNode
from .jaicp_weightnode import JAICP_WeightNode
from .jaicp_repeat import JAICP_Repeat
from .jaicp_wordnode import JAICP_WordNode
from .jaicp_named_pattern_with_label import JAICP_NamedPatternWithLabel
from .jaicp_negationnode import JAICP_NegationNode
from .jaicp_constituent import JAICP_Constituent


class JAICP_Pattern:
    def __init__(self, src_str, src_path=None):
        self.src_path = src_path
        self.src_str = src_str
        self.start_node = None
        self.weight_a = 1.0
        self.weight_b = 0.0
        self.converter = None
        self.check_count = 0

    def __repr__(self):
        return self.src_str

    @staticmethod
    def build(src_str, src_path, named_patterns):
        try:
            p = JAICP_Pattern(src_str, src_path=src_path)

            body_str = src_str
            converter_str = None

            if '||' in src_str:
                i = src_str.index('||')
                body_str = src_str[:i].strip()
                converter_str = src_str[i+2:].strip()

            p.start_node = JAICP_Pattern.join_nodes(JAICP_Pattern.build_sequence(body_str, named_patterns))
            p.converter = converter_str

            all_nodes = []
            p.start_node.list_all_nodes(all_nodes)
            for n in all_nodes:
                if isinstance(n, JAICP_WeightNode):
                    p.weight_a = n.a
                    p.weight_b = n.b
                    break

            return p
        except Exception as ex:
            msg = 'Exception on pattern {}: {}'.format(src_str, str(ex))
            raise ValueError(msg)

    def bind_named_patterns(self, named_patterns):
        """ Привязка именованных паттернов в узлах AST """
        self.start_node.bind_named_patterns(named_patterns)

    def bind_entities(self, entities):
        """ Привязка узлов AST к словарям сущностей """
        all_nodes = []
        self.start_node.list_all_nodes(all_nodes)
        for n in all_nodes:
            n.bind_entities(entities)
        # НАЧАЛО ОТЛАДКИ
        #if self.src_str == "((с/со/от) $DateTime (по/до) $DateTime)":
        #    print('DEBUG@72 {}'.format(len(all_nodes)))
        #    exit(0)
        # КОНЕЦ ОТЛАДКИ

    def optimize(self):
        #self.start_node = self.start_node.optimize()
        pass

    @staticmethod
    def build_next_node(tokens, token, named_patterns):
        if token == '*' or token.startswith('*{'):
            n = JAICP_StarNode(token)
        elif token == '$regexp':
            rx_str = tokens.read_rx_str()
            n = JAICP_RegexNode(rx_str)
        elif token == '$entity':
            ref_name = tokens.read_rx_str()
            n = JAICP_EntityNode(ref_name)
        elif token == '$morph':
            morph_str = tokens.read_rx_str()
            n = JAICP_MorphNode(morph_str)
        elif token == '$weight':
            w_str = tokens.read_rx_str()
            n = JAICP_WeightNode(w_str)
        elif token == '$repeat':
            n = JAICP_Repeat.parse(tokens, named_patterns)
        elif token == '$oneWord':
            n = JAICP_WordNode(token)
        elif token == '$nonEmptyGarbage':
            n = JAICP_WordNode(token)
        elif token[0] == '$':
            # Может быть маркировка вида $паттерн::метка
            if tokens.probe_read(':') and tokens.probe_read(':'):
                label = tokens.read()
            else:
                label = None

            return JAICP_NamedPatternWithLabel(token, label)
        else:
            n = JAICP_WordNode(token)

        return n

    @staticmethod
    def join_nodes(nodes):
        start = nodes[0]
        prev = start
        for node in nodes[1:]:
            prev.next_node = node
            prev = node
        return start

    @staticmethod
    def build_or_sequence_of_words(tokens):
        nodes = []
        while not tokens.eof():
            token = tokens.read()
            n = JAICP_WordNode(token)
            nodes.append(n)
            if not tokens.probe_read('/'):
                break

        return nodes

    @staticmethod
    def consume_1pattern(tokens, allow_or, named_patterns):
        pos0 = tokens.tell()
        t = tokens.read()

        if t == '(':
            str2 = tokens.read_tokens_untill_cparen()
            node = JAICP_Pattern.build_alternatives_node(str2, named_patterns)
        elif t == '{':
            str2 = tokens.read_tokens_untill_cparen()
            nx = JAICP_Pattern.build_sequence(str2, named_patterns)
            if len(nx) == 1:
                node = nx[0]
            else:
                node = JAICP_PermutationsNode.build(str2, nx)
        elif t == '[':
            str2 = tokens.read_tokens_untill_cparen()
            node = JAICP_OptionalNode(str2)
            node.node = JAICP_Pattern.join_nodes(JAICP_Pattern.build_sequence(str2, named_patterns))
        elif t == '^':
            raise NotImplementedError()
            arg_node = JAICP_Pattern.consume_1pattern(tokens, named_patterns=named_patterns)
            node = JAICP_NegationNode(str(arg_node), arg_node)
        else:
            node = JAICP_Pattern.build_next_node(tokens, t, named_patterns=named_patterns)

            if allow_or and tokens.probe('/'):
                tokens.seek(pos0)
                nx = JAICP_Pattern.build_or_sequence_of_words(tokens)
                node = JAICP_OrNode('/'.join(map(str, nx)))
                node.nodes = nx

        return node

    @staticmethod
    def build_sequence_from_tokens(tokens, named_patterns, allow_or):
        nodes = []

        while not tokens.eof():
            pos0 = tokens.tell()
            t = tokens.read()

            if t in ('/', '|'):
                if not allow_or:
                    tokens.seek(pos0)
                    break
                else:
                    raise RuntimeError()

            if t == ':':
                if not allow_or:
                    tokens.seek(pos0)
                    break
                else:
                    #raise RuntimeError()
                    label = tokens.read()
                    print('Unused markup: {}'.format(label))

            if t == '(':
                str2 = tokens.read_tokens_untill_cparen()
                node = JAICP_Pattern.build_alternatives_node(str2, named_patterns)
            elif t == '⟦':
                # Расширение движка JAICP: составляющая
                str2 = tokens.read_tokens_untill_cparen()
                node = JAICP_Constituent(str2)
            elif t == '{':
                str2 = tokens.read_tokens_untill_cparen()
                nx = JAICP_Pattern.build_sequence(str2, named_patterns)
                if len(nx) == 1:
                    node = nx[0]
                else:
                    node = JAICP_PermutationsNode.build(str2, nx)
            elif t == '[':
                str2 = tokens.read_tokens_untill_cparen()
                node = JAICP_OptionalNode(str2)
                node.node = JAICP_Pattern.build_alternatives_node(str2, named_patterns)
            elif t == '^':
                arg_node = JAICP_Pattern.consume_1pattern(tokens, named_patterns=named_patterns, allow_or=False)
                node = JAICP_NegationNode(str(arg_node), arg_node)
            else:
                node = JAICP_Pattern.build_next_node(tokens, t, named_patterns=named_patterns)

                if allow_or and (tokens.probe('/') or tokens.probe('|')):
                    # ??? надо ли так ???
                    raise NotImplementedError()
                    tokens.seek(pos0)
                    nx = JAICP_Pattern.build_or_sequence_of_words(tokens, named_patterns)
                    node = JAICP_OrNode('/'.join(map(str, nx)))
                    node.nodes = nx

            nodes.append(node)

        return nodes

    @staticmethod
    def build_sequence(src_str, named_patterns):
        try:
            tx = JAICP_Tokenizer.from_str(src_str)
            nodes = JAICP_Pattern.build_sequence_from_tokens(tx, named_patterns=named_patterns, allow_or=True)
            return nodes
        except Exception as ex:
            msg = 'Exception occured while parsing the pattern {}: {}'.format(src_str, str(ex))
            raise ValueError(msg)

    @staticmethod
    def load_alternatives(src_str, named_patterns):
        items = []

        tx = JAICP_Tokenizer.from_str(src_str)
        while not tx.eof():
            item_nodes = JAICP_Pattern.build_sequence_from_tokens(tx, named_patterns=named_patterns, allow_or=False)
            if item_nodes:
                item = JAICP_Pattern.join_nodes(item_nodes)
                if tx.probe_read(':'):
                    label = tx.read()
                else:
                    label = None

                items.append((item, label))

                if tx.probe('|') or tx.probe('/'):
                    tx.read()
                    continue
                else:
                    break
            else:
                raise RuntimeError()
                break

        return items

    @staticmethod
    def build_alternatives_node(src_str, named_patterns):
        alternatives = JAICP_Pattern.load_alternatives(src_str, named_patterns)
        if len(alternatives) == 1 and alternatives[0][0] is None:
            return alternatives[0][0]
        else:
            return JAICP_OrNode(src_str, alternatives)

    def calc_final_score(self, score):
        return score * self.weight_a + self.weight_b

    def match(self, words, cache, require_right_wall=True):
        """ Точка входа в процесс сопоставления запросной строки с одним глобальным паттерном. """
        mx = self.start_node.match(words, 0, cache)
        if mx:
            if require_right_wall:
                mx2 = []
                for m in mx:
                    if m.nb_words == len(words):
                        # Все токены запросной строки сопоставлены
                        mx2.append(m)
                    else:
                        # Если в запросе последний токен '?', то его можно (иногда нужно) не матчить
                        tail_ignore = True
                        for w in words.get_words(m.nb_words, len(words)-m.nb_words):
                            if w not in ('?', '.', '!'):
                                tail_ignore = False
                                break
                        if tail_ignore:
                            mx2.append(m)

                if mx2:
                    # Паттерн может дать несколько сопоставлений с разным весом, поэтому
                    # отсортируем и возьмем лучший вариант.
                    mx2 = sorted(mx2, key=lambda z: -z.score())
                    return mx2[0], self.calc_final_score(mx[0].score())
                else:
                    return None, 0.0
            else:
                mx = sorted(mx, key=lambda z: -z.score())
                return mx[0], self.calc_final_score(mx[0].score())

        return None, 0.0


if __name__ == '__main__':
    #ps = '* *{1} *стем* *{1,3} (кошка/собака) {много/*мало/конец*}'
    #ps = 'о/конец*'
    #ps = '(кошка/собака)'
    #ps = '[кош* лов* мыш*]'
    #ps = '{*кош* лов* мыш*}'
    #ps = '(кошка/собака/мышка)'
    #ps = '~кошка ~ловить ~мышка'
    #ps = '~кошка/*соб*/собакен* [уже/то/ведь] ~ловить/~поймать мыш*/крыс*'
    ps = '{кошка * мышку}'

    p = JAICP_Pattern.build(ps, src_path='TEST', named_patterns=dict())
    print(p)

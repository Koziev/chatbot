""" Представление узла AST для оператора $morph<...> """

import logging

from .matching import Matching
from .jaicp_basenode import JAICP_BaseNode


class JAICP_MorphNode(JAICP_BaseNode):
    def __init__(self, morph_str):
        src_str = '$morph<'+morph_str+'>'
        super(JAICP_MorphNode, self).__init__(src_str)
        self.morph_str = morph_str
        self.part_of_speech = None
        self.tags = []
        for t in morph_str.split():
            if t in ('С', 'NOUN'):
                self.part_of_speech = 'NOUN'
            elif t in ('П', 'ADJF', 'ADJ'):
                self.part_of_speech = 'ADJ'
            elif t in ('Г', 'VERB'):
                self.part_of_speech = 'VERB'
            elif t in ('ПРЕДЛ', 'ADP'):
                self.part_of_speech = 'PREP'
            elif t in ('Н', 'ADV'):
                self.part_of_speech = 'ADV'
            elif t in ('МС-П'):
                self.part_of_speech = 'ADJ'
            else:
                neg = False
                if t.startswith('^'):
                    neg = True
                    t = t[1:]

                if t in ('ед', 'sing', 'Sing'):
                    self.tags.append(('Number', 'Sing', neg))
                elif t in ('мн', 'plur', 'Plur'):
                    self.tags.append(('Number', 'Plur', neg))
                elif t in ('муж', 'masc', 'мр', 'Masc'):
                    self.tags.append(('Gender', 'Masc', neg))
                elif t in ('жен', 'femn', 'жр', 'Fem'):
                    self.tags.append(('Gender', 'Fem', neg))
                elif t in ('ср', 'neut', 'Neut'):
                    self.tags.append(('Gender', 'Neut', neg))
                elif t in ('им', 'nomn', 'Nom'):
                    self.tags.append(('Case', 'Nom', neg))
                elif t in ('род', 'gent', 'рд', 'Gen'):
                    self.tags.append(('Case', 'Gen', neg))
                elif t in ('твор', 'inst', 'Inst'):
                    self.tags.append(('Case', 'Ins', neg))
                elif t in ('вин', 'accs', 'вн', 'Acc'):
                    self.tags.append(('Case', 'Acc', neg))
                elif t in ('дат', 'datv', 'дт', 'Dat'):
                    self.tags.append(('Case', 'Dat', neg))
                elif t in ('предл', 'loct', 'loc2'):
                    self.tags.append(('Case', 'Loc', neg))
                elif t in ('инф', 'ИНФИНИТИВ', 'Inf'):
                    self.tags.append(('VerbForm', 'Inf', neg))
                elif t in ('ПРИЧАСТИЕ', 'Part'):
                    self.tags.append(('VerbForm', 'Part', neg))
                elif t in ('имп', 'Imp'):
                    self.tags.append(('VerbForm', 'Imp', neg))
                elif t == 'сравн':
                    self.tags.append(('Degree', 'Cmp', neg))
                elif t in ('нст', 'Pres'):
                    self.tags.append(('Tense', 'Pres', neg))
                elif t in ('буд', 'Fut'):
                    self.tags.append(('Tense', 'Fut', neg))
                elif t in ('КР_ПРИЛ', 'Short'):
                    self.tags.append(('Variant', 'Short', neg))
                elif t in ('Name', 'Ms-f', 'имя', 'фам', 'отч', 'мр-жр'):
                    logging.warning('Skipping tag %s in $morph<%s>', t, morph_str)
                else:
                    logging.error('Unknown tag "%s" in $morph<%s>'.format(t, morph_str))
                    exit(0)

    def __repr__(self):
        return '$morph<' + self.morph_str + '>' + self.get_next_node_repr()

    def match(self, words, start_index, cache):
        if start_index >= len(words):
            return []

        udp_token = words.get_token(start_index)  # +1
        if self.part_of_speech is not None:
            if udp_token.upos != self.part_of_speech:
                return []

        for tag_name, tag_value, is_neg in self.tags:
            v = udp_token.get_attr(tag_name)
            if v != tag_value and not is_neg:
                return []

            if v == tag_value and is_neg:
                return []

        mm1 = Matching()
        mm1.from_itoken = start_index
        mm1.nb_words = 1
        mm1.hits = 0.3

        if self.next_node:
            res_mx = []
            for m2 in self.next_node.match(words, start_index+1, cache):
                mm2 = Matching()
                mm2.from_itoken = start_index
                mm2.nb_words = mm1.nb_words + m2.nb_words
                mm2.inner_matchings = [mm1, m2]
                res_mx.append(mm2)

            return self.limit(res_mx)
        else:
            return [mm1]

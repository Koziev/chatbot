import os
import pickle
import math
import io

import networkx as nx


class RelevancyScore(object):
    def __init__(self, path, score):
        self.path = list(path)
        self.score = score

    def __repr__(self):
        return '[' + ' '.join(self.path) + '] {}'.format(self.score)

    def endlemmas(self):
        return set([self.path[0], self.path[-1]])


class RelevancyScorer(object):
    def __init__(self, parser):
        self.parser = parser
        self.score12_cache = dict()

    def load(self, model_dir):
        with open(os.path.join(model_dir, 'ruwordnet.pkl'), 'rb') as f:
            wordnet = pickle.load(f)

        self.TG = nx.Graph()
        for word1, words2 in wordnet.items():
            for word2 in words2:
                self.TG.add_edge(word1.replace('ё', 'е').lower(), word2.replace('ё', 'е').lower())

    def extract_lemmas(self, text):
        lemmas = set()
        for parsing in self.parser.parse_text(text):
            for t in parsing:
                if t.upos in ('VERB', 'ADJ', 'ADV', 'NOUN', 'PROPN'):
                    lemmas.add(t.lemma)
        return lemmas

    def score_relevancy(self, premise, query):
        p_lemmas = self.extract_lemmas(premise)
        q_lemmas = self.extract_lemmas(query)
        return self._score(p_lemmas, q_lemmas)

    def _score(self, p_lemmas, q_lemmas):
        min_path = None
        for q_lemma in q_lemmas:
            for p_lemma in p_lemmas:
                if q_lemma == p_lemma:
                    min_path = [q_lemma]
                    break

                if p_lemma in self.TG and q_lemma in self.TG:
                    try:
                        p = nx.shortest_path(self.TG, source=q_lemma, target=p_lemma)
                        if min_path is None or len(p) < len(min_path):
                            min_path = p
                    except nx.exception.NetworkXNoPath as ex:
                        pass

        if min_path is None:
            return RelevancyScore([], 0.0)
        else:
            return RelevancyScore(min_path, math.exp(-(len(min_path)-1)*0.5))

    def match1(self, query, facts, threshold=0.3):
        matches = []
        q_lemmas = self.extract_lemmas(query)
        for premise, _, _ in facts:
            p_lemmas = self.extract_lemmas(premise)
            score = self._score(p_lemmas, q_lemmas)
            if score.score >= threshold:
                matches.append((premise, score))

        matches = sorted(matches, key=lambda z: -z[1].score)
        return matches

    def match2(self, query, facts, threshold=0.3):
        matches1 = []
        premise2score = dict()

        q_lemmas = self.extract_lemmas(query)
        for premise, _, _ in facts:
            p_lemmas = self.extract_lemmas(premise)
            score = self._score(p_lemmas, q_lemmas)
            if score.score >= threshold:
                matches1.append((premise, score))
                premise2score[premise] = score

        matches1 = sorted(matches1, key=lambda z: -z[1].score)
        matches2 = []
        matched_pairs = set()
        for premise1, score1 in matches1[:10]:
            for premise2, _, _ in facts:
                if premise1 != premise2 and premise2 in premise2score:
                    # НАЧАЛО ОТЛАДКИ
                    #if premise1 == 'Сократ - философ' and premise2 == 'все философы смертны':
                    #    print('DEBUG@101')
                    # КОНЕЦ ОТЛАДКИ

                    score2 = premise2score[premise2]  # premise2 <==> query
                    if not any((lemma in score1.endlemmas()) for lemma in score2.endlemmas()):
                        if (premise1, premise2) not in matched_pairs and (premise2, premise1) not in matched_pairs:
                            score12 = self.score12_cache.get((premise1, premise2))
                            if score12 is None:
                                score12 = self.score_relevancy(premise1, premise2)  # premise1 <==> premise2
                                self.score12_cache[(premise1, premise2)] = score12

                            if score12.score >= threshold:
                                score12_endlemmas = score12.endlemmas()
                                if not any((lemma in score1.endlemmas()) for lemma in score12_endlemmas):
                                    if not any((lemma in score2.endlemmas()) for lemma in score12_endlemmas):
                                        total_score = score1.score * score2.score * score12.score
                                        matches2.append((premise1, premise2, total_score))
                                        matched_pairs.add((premise1, premise2))

        matches2 = sorted(matches2, key=lambda z: -z[2])
        return matches2


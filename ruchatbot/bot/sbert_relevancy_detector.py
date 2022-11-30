"""
Обертка для модели определения релевантности контекста и вопроса (premise-question relevancy)
с архитектурой Sentence Transformer.
"""

import sentence_transformers
from ruchatbot.bot.search_utils import normalize_for_lookup


class SbertRelevancyDetector(object):
    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self, model_dir):
        self.model = sentence_transformers.SentenceTransformer(model_dir, device=self.device)

    def calc_relevancy1(self, premise, query, **kwargs):
        embeddings = self.model.encode([premise, query])
        y = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
        return y

    def get_most_relevant(self, query, premises, nb_results=1):
        # 30.11.2022 иногда происходят поиски фраз, которые фактически совпадают с одним из фактов, до регистра.
        # Можно немного улучшить производительность для таких случаев, пройдясь по списку и сделав строковое сравнение.
        uquery = normalize_for_lookup(query)
        for premise in premises:
            if uquery == normalize_for_lookup(premise[0]):
                return [premise[0]], [1.0]

        embeddings = self.model.encode([p[0] for p in premises] + [query], convert_to_tensor=True, device=self.device)
        q1_v = embeddings[-1].unsqueeze(dim=0)
        px_v = embeddings[:-1]
        rx = sentence_transformers.util.semantic_search(query_embeddings=q1_v, corpus_embeddings=px_v,
                                                        query_chunk_size=100, corpus_chunk_size=100, top_k=nb_results)

        closest_premises = [(premises[x['corpus_id']][0], x['score']) for x in rx[0] if x['score'] >= 0.70]
        closest_premises = sorted(closest_premises, key=lambda z: -z[1])

        return [x[0] for x in closest_premises], [x[1] for x in closest_premises]

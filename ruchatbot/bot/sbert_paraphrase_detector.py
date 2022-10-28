"""
Обертка для модели определения синонимичности двух коротких текстов с архитектурой Sentence Transformer.
"""

import sentence_transformers


class SbertSynonymyDetector(object):
    def __init__(self, device):
        self.device = device
        self.model = None

    def load(self, model_dir):
        self.model = sentence_transformers.SentenceTransformer(model_dir, device=self.device)

    def calc_synonymy1(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        y = sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1])
        return y

    def get_most_similar(self, probe_phrase, phrases, nb_results=1):
        embeddings = self.model.encode([p[0] for p in phrases]  + [probe_phrase], convert_to_tensor=True, device=self.device)
        q1_v = embeddings[-1].unsqueeze(dim=0)
        px_v = embeddings[:-1]
        rx = sentence_transformers.util.semantic_search(query_embeddings=q1_v, corpus_embeddings=px_v,
                                                        query_chunk_size=100, corpus_chunk_size=100, top_k=nb_results)

        closest_premises = [(phrases[x['corpus_id']][0], x['score']) for x in rx[0] if x['score'] >= 0.70]
        closest_premises = sorted(closest_premises, key=lambda z: -z[1])

        return [x[0] for x in closest_premises], [x[1] for x in closest_premises]

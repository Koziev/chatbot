class PhraseToken:
    def __init__(self):
        self.word = None
        self.norm_word = None
        self.lemma = None
        self.word_index = None
        self.chunk_index = None
        self.tagset = None
        self.is_chunk_starter = None

    def __repr__(self):
        return self.word

    def is_noun(self):
        return self.tagset.startswith('NOUN')

    def is_inf(self):
        return self.tagset.startswith('VERB') and 'VerbForm=Inf' in self.tagset

    def is_verb(self):
        return self.tagset.startswith('VERB')

    def is_adj(self):
        return self.tagset.startswith('ADJ')

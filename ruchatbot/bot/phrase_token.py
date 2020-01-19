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

class Chunk:
    def __init__(self):
        self.text = None
        self.tokens = None
        self.head = None
        self.core_text = None

    def __repr__(self):
        return self.text

    def contains_token(self, token_id):
        return any(z.id == token_id for z in self.tokens)

class Matching:
    def __init__(self):
        self.from_itoken = -1
        self.nb_words = 0
        self.hits = 0.0
        self.penalty = 0.0
        self.inner_matchings = []
        self.label = None
        self.entity_item = None
        self.matcher = None

    def score(self):
        return self.hits - self.penalty + sum(z.score() for z in self.inner_matchings)

    def extract_markup(self, parsing, name2value):
        if self.label is not None:
            words = parsing.get_words(self.from_itoken, self.nb_words)
            name2value[self.label] = words
        if self.inner_matchings:
            for m in self.inner_matchings:
                m.extract_markup(parsing, name2value)

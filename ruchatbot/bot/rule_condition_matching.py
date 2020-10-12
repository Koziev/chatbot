class RuleConditionMatchGroup:
    def __init__(self, name, words, phrase_tokens):
        self.name = name
        self.words = words
        self.phrase_tokens = phrase_tokens

    def __repr__(self):
        s = self.name
        s += ' = '
        s += ' '.join(self.words)
        return s


class RuleConditionMatching(object):
    def __init__(self):
        self.success = False
        self.proba = 0.0
        self.groups = dict()  # str -> RuleConditionMatchGroup()

    def __repr__(self):
        return ' '.join(map(str, self.groups))

    @staticmethod
    def create(success):
        r = RuleConditionMatching()
        r.success = success
        r.proba = 1.0 if success else 0.0
        return r

    def add_group(self, name, words, phrase_tokens):
        g = RuleConditionMatchGroup(name, words, phrase_tokens)
        self.groups[name] = g

    def has_groups(self):
        return self.groups

import copy


class MatchingCache:
    def __init__(self):
        self.pos_name2matchings = dict()
        self.pos_id2matchings = dict()

    def get(self, itoken, pattern_name):
        return self.pos_name2matchings.get((itoken, pattern_name), None)

    def store(self, itoken, pattern_name, matchings):
        self.pos_name2matchings[(itoken, pattern_name)] = [copy.deepcopy(x) for x in matchings]

    def get_by_id(self, itoken, node_id):
        return self.pos_id2matchings.get((itoken, node_id), None)

    def store_by_id(self, itoken, node_id, matchings):
        self.pos_id2matchings[(itoken, node_id)] = [copy.deepcopy(x) for x in matchings]

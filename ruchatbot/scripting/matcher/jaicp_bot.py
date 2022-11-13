import logging
import pickle

from .load_named_patterns import load_named_patterns
from .load_global_rules import load_global_rules
from .jaicp_entities import load_all_entities
from .matching_cache import MatchingCache


class JAICP_Bot:
    def __init__(self):
        self.entities = dict()
        self.named_patterns = dict()
        self.global_rules = []
        self.parser = None

    def load_all_entities(self, dirs: [str]):
        self.entities = load_all_entities(dirs)

    def load_named_patterns(self, patterns_dirs: [str]):
        self.named_patterns = load_named_patterns(patterns_dirs, self.entities)
        logging.info('%d named patterns loaded in total', len(self.named_patterns))

    def load_global_rules(self, patterns_dirs: [str]):
        self.global_rules = load_global_rules(patterns_dirs, self.named_patterns, self.entities)
        logging.info('%d global rules loaded in total', len(self.global_rules))

    def match(self, query_parsing, top_n: int):
        cache = MatchingCache()
        res = list()
        for global_rule in self.global_rules:
            try:
                matching, score = global_rule.pattern.match(query_parsing, cache)
                if matching is not None and score > 0.0:
                    res.append((global_rule, matching, score))
            except:
                logging.error('Error occured when executing global rule with pattern "%s"', global_rule.pattern)
                raise

        return sorted(res, key=lambda z: -z[2])[:top_n]

    def serialize(self, p: str):
        with open(p, 'wb') as f:
            pickle.dump(self.entities, f)
            pickle.dump(self.named_patterns, f)
            pickle.dump(self.global_rules, f)

    def deserialize(self, p: str):
        with open(p, 'rb') as f:
            self.entities = pickle.load(f)
            self.named_patterns = pickle.load(f)
            self.global_rules = pickle.load(f)

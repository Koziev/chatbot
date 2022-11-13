import itertools

from ruchatbot.bot.conversation_engine import DialogHistory
from ruchatbot.scripting.matcher.jaicp_pattern import JAICP_Pattern
from ruchatbot.utils.udpipe_parser import Parsing
from ruchatbot.scripting.actors import ActorBase


class DialogRuleMatching(object):
    def __init__(self, matched_rule, total_score, matchings, action_results):
        self.matched_rule = matched_rule
        self.total_score = total_score
        self.matchings = list(matchings)
        self.action_results = list(action_results)

    def __repr__(self):
        s = 'score:{:5.3f} ==> {}'.format(self.total_score, ' | '.join(map(str, self.action_results)))
        return s

    def get_actions(self):
        return self.action_results


class DialogRule(object):
    def __init__(self):
        self.name = None
        self.patterns = []  # list[JAICP_Pattern]
        self.actors = []

    def __repr__(self):
        if self.name:
            s = self.name + ': '
        else:
            s = ''

        s += 'if {} then {}'.format(' | '.join(map(str, self.patterns)), ' '.join(map(str, self.actors)))
        return s

    @staticmethod
    def load_from_yaml(yaml_node, constants, named_patterns, entities, text_utils):
        rule = DialogRule()
        if 'name' in yaml_node['rule']:
            rule.name = yaml_node['rule']['name']

        pattern_str = yaml_node['rule']['if']['q']

        pattern = JAICP_Pattern.build(pattern_str, named_patterns=named_patterns, src_path='<<<UNKNOWN>>>')
        pattern.bind_named_patterns(named_patterns)
        pattern.bind_entities(entities)
        pattern.optimize()

        rule.patterns.append(pattern)

        if isinstance(yaml_node['rule']['then'], dict):
            actor = ActorBase.load_from_yaml(yaml_node['rule']['then'], constants, text_utils)
            rule.actors.append(actor)
        else:
            raise NotImplementedError()

        return rule

    def match(self, dialog_history, parsing_cache, matching_cache, session, text_utils):
        if len(dialog_history) < len(self.patterns):
            # история диалога недостаточно длинная
            return None

        matchings = []
        total_score = 1.0
        for ipattern, pattern in enumerate(self.patterns):
            utterance_text = dialog_history.messages[-ipattern-1].get_text()
            if utterance_text in parsing_cache:
                parsing = parsing_cache[utterance_text]
            else:
                utterance_parsings = text_utils.parser.parse_text(utterance_text)
                parsing = Parsing(tokens=itertools.chain(*utterance_parsings), text=utterance_text)
                parsing_cache[utterance_text] = parsing

            matching, score = pattern.match(parsing, matching_cache)
            if matching is None or score == 0.0:
                return None

            matchings.append(matching)
            total_score *= score

        results = [actor.do_action(matching, session, text_utils) for actor in self.actors]
        results2 = list(itertools.chain(*results))
        return DialogRuleMatching(self, total_score, matchings, results2)

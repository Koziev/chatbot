import itertools

from ruchatbot.bot.conversation_engine import DialogHistory
from ruchatbot.scripting.matcher.jaicp_pattern import JAICP_Pattern
#from ruchatbot.utils.udpipe_parser import Parsing
from ruchatbot.scripting.actors import ActorBase
from ruchatbot.scripting.matcher.parsing_result import ParsingResult


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
        self.condition_keyword = None
        self.patterns = []  # list[JAICP_Pattern]
        self.actors = []

    def __repr__(self):
        if self.name:
            s = self.name + ': '
        else:
            s = ''

        s += 'if {}: {} then {}'.format(self.condition_keyword, ' | '.join(map(str, self.patterns)), ' '.join(map(str, self.actors)))
        return s

    @staticmethod
    def load_from_yaml(yaml_node, constants, named_patterns, entities, generative_named_patterns, text_utils):
        rule = DialogRule()
        if 'name' in yaml_node['rule']:
            rule.name = yaml_node['rule']['name']

        rule.condition_keyword = list(yaml_node['rule']['if'].keys())[0]

        if rule.condition_keyword == 'h':
            # Для обычных правил, описывающих реакцию на последнюю реплику человека
            pattern_str = yaml_node['rule']['if'][rule.condition_keyword]
            pattern = JAICP_Pattern.build(pattern_str, named_patterns=named_patterns, src_path='<<<UNKNOWN>>>')
            pattern.bind_named_patterns(named_patterns)
            pattern.bind_entities(entities)
            pattern.optimize()
            rule.patterns.append(('h1', pattern))
        elif rule.condition_keyword in ('hb', 'bh', 'hbh'):
            keys = []
            if rule.condition_keyword == 'hb':
                keys = ['h2', 'b1']
            elif rule.condition_keyword == 'bh':
                # предпоследняя реплика - бот
                # последняя реплика - человек
                keys = ['b2', 'h1']
            elif rule.condition_keyword == 'hbh':
                keys = ['h3', 'b2', 'h1']
            else:
                raise NotImplementedError()

            for reply_key, pattern_str in zip(keys, yaml_node['rule']['if'][rule.condition_keyword]):
                pattern = JAICP_Pattern.build(pattern_str, named_patterns=named_patterns, src_path='<<<UNKNOWN>>>')
                pattern.bind_named_patterns(named_patterns)
                pattern.bind_entities(entities)
                pattern.optimize()
                rule.patterns.append((reply_key, pattern))
        else:
            raise NotImplementedError()

        if isinstance(yaml_node['rule']['then'], dict):
            actor = ActorBase.load_from_yaml(yaml_node['rule']['then'], constants, generative_named_patterns, text_utils)
            rule.actors.append(actor)
        else:
            raise NotImplementedError()

        return rule

    def match(self, dialog_context, parsing_cache, matching_cache, session, text_utils):
        if len(dialog_context) < len(self.patterns):
            # история диалога недостаточно длинная
            return None

        matchings = []
        total_score = 1.0
        for ipattern, (reply_key, pattern) in enumerate(self.patterns):
            if reply_key not in dialog_context:
                return None

            utterance = dialog_context[reply_key]
            best_score = 0.0
            best_matching = None
            # пробуем сопоставить паттерн правила с исходным текстом реплики и с раскрытым текстом.
            # оставим сопоставление с максимальным скором.
            for utterance_text in [utterance.get_text(), utterance.get_interpretation()]:
                if utterance_text:
                    if utterance_text in parsing_cache:
                        parsing = parsing_cache[utterance_text]
                    else:
                        utterance_parsings = text_utils.parser.parse_text(utterance_text)
                        #parsing = Parsing(tokens=itertools.chain(*utterance_parsings), text=utterance_text)
                        parsing = ParsingResult(tokens=itertools.chain(*utterance_parsings), text=utterance_text)
                        parsing_cache[utterance_text] = parsing

                    matching, score = pattern.match(parsing, matching_cache)
                    if matching is not None and score > 0.0 and score > best_score:
                        best_score = score
                        best_matching = matching

            if best_matching is None:
                return None

            matchings.append(best_matching)
            total_score *= best_score

        results = [actor.do_action(matching, session, text_utils) for actor in self.actors]
        results = [result for result in results if result]
        if results:
            results2 = list(itertools.chain(*results))
            return DialogRuleMatching(self, total_score, matchings, results2)
        else:
            # Такое может быть, если оператор say не смог сгенерировать ответ.
            # В таком случае считаем, что правило не подошло.
            return None
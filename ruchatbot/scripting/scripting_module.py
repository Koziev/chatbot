"""
Именованные модули - группы правил, которые можно импортировать в сценариях по имени.
"""

from ruchatbot.scripting.dialog_rule import DialogRule
from ruchatbot.scripting.matcher.jaicp_pattern import JAICP_Pattern
from ruchatbot.scripting.generator.generative_template import TemplatePattern


class ScriptingModule(object):
    def __init__(self):
        self.name = None
        self.greedy_rules = []
        self.rewrite_rules = []
        self.named_patterns = dict()
        self.named_generators = dict()

    def get_name(self):
        return self.name

    @staticmethod
    def load_from_yaml(yaml_node, modules, constants, named_patterns, entities, generative_named_patterns, text_utils):
        module = ScriptingModule()
        module.name = yaml_node['name']
        try:
            if 'greedy_rules' in yaml_node:
                for rule in yaml_node['greedy_rules']:
                    rule = DialogRule.load_from_yaml(rule, constants, named_patterns, entities, generative_named_patterns, text_utils)
                    module.greedy_rules.append(rule)

            if 'rewrite_rules' in yaml_node:
                for rule in yaml_node['rewrite_rules']:
                    rule = DialogRule.load_from_yaml(rule, constants, named_patterns, entities, generative_named_patterns, text_utils)
                    module.rewrite_rules.append(rule)

            if 'patterns' in yaml_node:
                for pattern_name, pattern_str in yaml_node['patterns'].items():
                    all_named_patterns = {**named_patterns, **module.named_patterns}
                    pattern = JAICP_Pattern.build(pattern_str, named_patterns=all_named_patterns, src_path='<<<UNKNOWN>>>')
                    pattern.bind_named_patterns(named_patterns)
                    pattern.bind_entities(entities)
                    pattern.optimize()
                    module.named_patterns[pattern_name] = pattern

            if 'generators' in yaml_node:
                for generator_name, generator_str in yaml_node['generators'].items():
                    all_named_generators = {**generative_named_patterns, **module.named_generators}
                    generator = TemplatePattern(pattern_str, all_named_generators)
                    module.named_generators[generator_name] = generator

        except Exception as ex:
            print('Error occured in module "{}" body parsing:\n{}'.format(module.name, str(ex)))
            raise

        return module

"""
Именованные модули - группы правил, которые можно импортировать в сценариях по имени.
"""

from ruchatbot.scripting.dialog_rule import DialogRule


class ScriptingModule(object):
    def __init__(self):
        self.name = None
        self.greedy_rules = []
        self.rewrite_rules = []

    def get_name(self):
        return self.name

    @staticmethod
    def load_from_yaml(yaml_node, modules, constants, named_patterns, entities, text_utils):
        module = ScriptingModule()
        module.name = yaml_node['name']
        try:
            if 'greedy_rules' in yaml_node:
                for rule in yaml_node['greedy_rules']:
                    rule = DialogRule.load_from_yaml(rule, constants, named_patterns, entities, text_utils)
                    module.greedy_rules.append(rule)

            if 'rewrite_rules' in yaml_node:
                for rule in yaml_node['rewrite_rules']:
                    rule = DialogRule.load_from_yaml(rule, constants, named_patterns, entities, text_utils)
                    module.rewrite_rules.append(rule)

        except Exception as ex:
            print('Error occured in module "{}" body parsing:\n{}'.format(module.name, str(ex)))
            raise

        return module

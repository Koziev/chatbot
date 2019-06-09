# -*- coding: utf-8 -*-

import yaml


class ComprehensionTable(object):
    def __init__(self):
        self.templates = []
        self.order2anchor = dict()

    @staticmethod
    def __get_node_list(node):
        if isinstance(node, list):
            return node
        else:
            return [node]

    def load_yaml_data(self, yaml_data):
        """
        Из YAML конфига загружаются правила трансляции.
        Одно правило в простейшем случае описывает ситуацию, когда одна входная
        фраза преобразуется в новую фразу. Например, входная фраза:

        - Ты должен назвать свое имя.

        будет преобразована в

        - Как тебя зовут?

        В общем случае, одно правило может задавать несколько альтернативных входных фраз,
        чтобы компактно задавать их трансляцию.
        """
        if 'comprehensions' in yaml_data:
            for rule in yaml_data['comprehensions']:
                result_phrase = rule['rule']['then']
                conditions = ComprehensionTable.__get_node_list(rule['rule']['if'])
                for input_phrase in conditions:
                    self.templates.append((result_phrase, input_phrase))
                    self.order2anchor[input_phrase] = result_phrase

    def get_templates(self):
        return list(self.templates)

    def get_order_anchor(self, order):
        return self.order2anchor.get(order, None)

# -*- coding: utf-8 -*-

from ruchatbot.bot.actors import ActorBase


class VerbalFormField(object):
    def __init__(self):
        self.name = None
        self.from_entity = None  # имя entity для заполнения
        self.from_reflection = None  # текст вопроса, который бот задаст себе, для поиска ответа
        self.question = None  # текст вопроса к пользователю


class VerbalForm(object):
    def __init__(self):
        self.name = None
        self.fields = []
        self.ok_action = None  # действие при успешном заполнении всех полей
        self.compiled_ok_action = None
        self.insteadof_rules = None  # особые правила обработки ответов
        self.smalltalk_rules = None

    @staticmethod
    def from_yaml(yaml_node):
        form = VerbalForm()
        form.name = yaml_node['name']
        form.ok_action = yaml_node['action']
        form.compiled_ok_action = ActorBase.from_yaml(form.ok_action)

        if 'fields' in yaml_node:
            for field_node in yaml_node['fields']:
                field = VerbalFormField()
                field_node = field_node['field']
                field.name = field_node['name']
                field.question = field_node['question']
                if 'from_entity' in field_node:
                    field.from_entity = field_node['from_entity']
                if 'from_reflection' in field_node:
                    field.from_reflection = field_node['from_reflection']
                form.fields.append(field)

        # TODO: сделать загрузку instead-of и smalltalk правил
        #self.insteadof_rules

        return form

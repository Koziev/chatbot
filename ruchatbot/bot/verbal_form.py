# -*- coding: utf-8 -*-

import logging

from ruchatbot.bot.actors import ActorBase
from ruchatbot.utils.constant_replacer import replace_constant


class VerbalFormField(object):
    """Описание одного поля в форме"""
    def __init__(self):
        self.name = None
        self.source = None  # будет ли слот заполнен результатом NER или сырым ответом пользователя
        self.from_entity = None  # имя entity для заполнения
        self.from_reflection = None  # текст вопроса, который бот задаст себе, для поиска ответа
        self.question = None  # текст вопроса к пользователю


class VerbalForm(object):
    """Вербальная форма - формализованная сценарий ввода набора данных пользователем"""
    def __init__(self):
        self.name = None
        self.fields = []
        self.ok_action = None  # действие при успешном заполнении всех полей
        self.compiled_ok_action = None
        self.insteadof_rules = None  # особые правила обработки ответов
        self.smalltalk_rules = None

    def get_name(self):
        return self.name

    @staticmethod
    def from_yaml(yaml_node, constants, text_utils):
        form = VerbalForm()
        form.name = yaml_node['name']
        form.ok_action = yaml_node['action']
        form.compiled_ok_action = ActorBase.from_yaml(form.ok_action, constants, text_utils)

        if 'fields' in yaml_node:
            for field_node in yaml_node['fields']:
                field = VerbalFormField()
                field_node = field_node['field']
                field.name = field_node['name']
                field.question = replace_constant(field_node['question'], constants, text_utils)
                if 'from_entity' in field_node:
                    field.from_entity = field_node['from_entity']
                    field.source = 'entity'
                elif 'from_reflection' in field_node:
                    field.from_reflection = field_node['from_reflection']
                    field.source = 'reflection'
                elif 'source' in field_node:
                    field.source = field_node['source']
                    if field.source not in ('raw_response', 'entity', 'reflection'):
                        logging.error(u'Unknown field source "%s"', field.source)
                        raise RuntimeError()
                else:
                    # TODO: сделать внятную диагностику
                    raise NotImplementedError()

                form.fields.append(field)

        # TODO: сделать загрузку instead-of и smalltalk правил, которые будут проверяться при работе форму.
        #self.insteadof_rules

        return form

# -*- coding: utf-8 -*-


class LanguageResources:
    """
    Экземпляр класса выступает в роли хранилища текста базовых реплик робота
    типа ДА или НЕТ, чтобы не размазывать по коду локализуемые строки.
    """
    def __init__(self):
        self.key2phrase = dict()

    def __getitem__(self, key_phrase):
        # по умолчанию возвращается текст запрошенной фразы без изменений,
        # то есть для u'да' возвращается u'да', и т.д.
        return self.key2phrase.get(key_phrase, key_phrase)

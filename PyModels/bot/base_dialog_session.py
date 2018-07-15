# -*- coding: utf-8 -*-


class BaseDialogSession(object):
    """
    Хранилище оперативных данных для диалоговой сессии с одним собеседником.
    Персистентность реализуется производными классами.
    """
    def __init__(self, interlocutor, facts_storage):
        """
        Инициализация новой диалоговой сессии для нового собеседника.
        :param interlocutor: строковый идентификатор собеседника.
        :param facts_storage: объект класса BaseFactsStorage, реализующего
         чтение и сохранение фактов.
        """
        self.interlocutor = interlocutor
        self.facts_storage = facts_storage
        self.answer_buffer = []

    def add_to_buffer(self, phrase):
        """
        В буфер ожидающих выдачи ответов бота добавляем новую реплику
        :param phrase: добавляемая реплика
        """
        assert(phrase is not None and len(phrase) > 0)
        self.answer_buffer.append(phrase)

    def extract_from_buffer(self):
        """
        Извлекает и возвращает самую старую готовую фразу
        из буфера ответов.
        :return: ответ бота или пустая строка, если буфер ответов пуст.
        """
        if len(self.answer_buffer)==0:
            return u''

        return self.answer_buffer.pop(0)

    def store_new_fact(self, fact):
        """
        К списку фактов добавляется новый, полученный в результате
        диалога с пользователем. В зависимости от реализации хранилища
        факт может быть запомнен либо только в памяти, либо сохранен
        в файлах, БД etc
        :param fact:
        """
        self.facts_storage.store_new_fact(interlocutor=self.interlocutor, fact=fact)

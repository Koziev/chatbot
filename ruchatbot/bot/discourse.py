"""
Фоновый процесс обработки беседы, сохраняющий в базе знаний всякие сведения о дискурсе

TODO: сейчас в этом коде много русскоязычных шаблонов вбито хардкодом. Потом надо будет вынести
текст шаблонов в конфиг бота.
"""


class Discourse:
    def __init__(self):
        pass

    def process_interrogator_phrase(self, bot, session, interpreted_phrase):
        """тут будет извлечение темы беседы и запоминание в базе данных"""
        pass

    def process_bot_phrase(self, bot, session, phrase):
        """тут будет извлечение темы беседы и запоминание в базе данных"""
        pass

    def store_assertion_in_database(self, bot, session, interpreted_phrase):
        fact_text = 'ты сказал, что {}'.format(interpreted_phrase.interpretation)
        fact = (fact_text, '3', 'last_interrogator_assertion')
        bot.facts.store_new_fact(session.get_interlocutor(), fact, unique=True)

    def store_question_in_database(self, bot, session, interpreted_phrase):
        fact_text = 'ты спросил: {}'.format(interpreted_phrase.interpretation)
        fact = (fact_text, '3', 'last_interrogator_question')
        bot.facts.store_new_fact(session.get_interlocutor(), fact, unique=True)

    def store_order_in_database(self, bot, session, interpreted_phrase):
        self.store_assertion_in_database(bot, session, interpreted_phrase)

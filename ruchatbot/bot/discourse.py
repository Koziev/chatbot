"""
Фоновый процесс обработки беседы, сохраняющий в базе знаний всякие сведения о дискурсе

TODO: сейчас в этом коде много русскоязычных шаблонов вбито хардкодом. Потом надо будет вынести
текст шаблонов в конфиг бота.

01.01.2021 Поправлена текстовка факта, добавляемого в базу знаний при вводе императива
01.01.2021 Учитываем пол собеседника в тексте фактов, добавляемых в базу при вводе утверждения, императива, вопроса
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

    def get_interlocutor_gender(self, bot, interlocutor):
        interlocutor_gender = bot.facts.find_tagged_fact(interlocutor, bot.facts.INTERCOLUTOR_GENDER_FACT)
        if interlocutor_gender:
            if interlocutor_gender == 'ты женского пола':
                interlocutor_gender = 'Fem'
            else:
                interlocutor_gender = 'Masc'
        else:
            interlocutor_gender = 'Masc'
        return interlocutor_gender

    def store_assertion_in_database(self, bot, session, interpreted_phrase):
        interlocutor_gender = self.get_interlocutor_gender(bot, session.get_interlocutor())
        if interlocutor_gender == 'Fem':
            fact_text = 'ты сказала, что {}'.format(interpreted_phrase.interpretation)
        else:
            fact_text = 'ты сказал, что {}'.format(interpreted_phrase.interpretation)

        fact = (fact_text, '3', 'last_interrogator_assertion')
        bot.facts.store_new_fact(session.get_interlocutor(), fact, unique=True)

    def store_question_in_database(self, bot, session, interpreted_phrase):
        interlocutor_gender = self.get_interlocutor_gender(bot, session.get_interlocutor())
        if interlocutor_gender == 'Fem':
            fact_text = 'ты спросила: {}'.format(interpreted_phrase.interpretation)
        else:
            fact_text = 'ты спросил: {}'.format(interpreted_phrase.interpretation)

        fact = (fact_text, '3', 'last_interrogator_question')
        bot.facts.store_new_fact(session.get_interlocutor(), fact, unique=True)

    def store_order_in_database(self, bot, session, interpreted_phrase):
        interlocutor_gender = self.get_interlocutor_gender(bot, session.get_interlocutor())
        if interlocutor_gender == 'Fem':
            fact_text = 'ты сказала: {}'.format(interpreted_phrase.interpretation)
        else:
            fact_text = 'ты сказал: {}'.format(interpreted_phrase.interpretation)

        fact = (fact_text, '3', 'last_interrogator_order')
        bot.facts.store_new_fact(session.get_interlocutor(), fact, unique=True)

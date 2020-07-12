
import random

from ruchatbot.bot.scenario import Scenario
from ruchatbot.bot.running_scenario import RunningScenario
from ruchatbot.utils.constant_replacer import replace_constant


class Scenario_WhoAmI(Scenario):
    def __init__(self):
        super(Scenario_WhoAmI, self).__init__()

        # название сценария используется в правилах для активации сценария.
        self.name = "кто_я"

        # приоритет определяет порядок вытеснения дних сценариев другими.
        self.priority = 48

        # Добавим счетчик количества запуска этого сценария, чтобы реагировать
        # на повторные вопросы "Кто ты?"
        self.activation_counter = 0

        # Сколько раз бот попытался узнать у собеседника, кто он такой
        self.who_are_you_counter = 0

    def can_process_questions(self):
        return True

    def started(self, running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils):
        # Активация сценария.
        self.activation_counter += 1

    def are_similar(self, bot, text_utils, phrase1, phrase2):
        syn = bot.get_engine().get_synonymy_detector()
        sim = syn.calc_synonymy2(phrase1, phrase2, text_utils)
        return sim >= syn.get_threshold()

    def is_similar_any(self, bot, text_utils, phrase1, phrases2_0):
        syn = bot.get_engine().get_synonymy_detector()
        phrases2 = [(phrase, None, None) for phrase in phrases2_0]
        best_phrase, best_sim = syn.get_most_similar(phrase1, phrases2, text_utils, nb_results=1)
        return best_sim >= syn.get_threshold()

    def process_question(self, running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils):
        # Обычно вопросы внутри диалога не обрабатываются. Но для более живого диалога это полезно.
        sx = ['тебе-то что',
              'тебе-то какое дело'
              'почему я должен отвечать на этот вопрос',
              'зачем тебе это',
              'почему ты спрашиваешь об этом',
              'почему ты интересуешься этим',
              'зачем тебе эта информация',
              'неужели тебе это интересно',
              'неужто ты хочешь знать это',
              'в смысле',
              'о чем речь',
              'что ты имеешь в виду',
              'ты хочешь знать, кто я']
        if self.is_similar_any(bot, text_utils, interpreted_phrase.raw_phrase, sx):
            sx = ['Я просто хочу понимать, с кем разговариваю',
                  'Хочу узнать, чем ты занимаешься',
                  'Мне хотелось бы узнать, чем ты занимаешься',
                  'Хочу познакомиться с тобой',
                  'Хочу узнать о тебе']
            bot.say(session, self.choice(bot, session, text_utils, sx))
            return True

        return False

    def choice(self, bot, session, text_utils, phrases0):
        # Подстановки констант (имя бота etc)
        phrases = [replace_constant(phrase, bot.profile.constants, text_utils) for phrase in phrases0]

        if len(phrases) == 1:
            return phrases[0]

        phrases2 = list(filter(lambda z: session.count_bot_phrase(z) == 0, phrases))
        if len(phrases2) == 0:
            phrases2 = phrases

        return random.choice(phrases2)

    def run_step(self, running_scenario, bot, session, interlocutor, interpreted_phrase, text_utils):
        running_scenario.current_step_index += 1

        if running_scenario.current_step_index == 0:
            # Первый шаг. Пользователь спросил "Кто ты?".
            # Пусть движок выдаст ответ на вопрос "Кто я" из своей базы знаний.
            # Это более гибкий подход, нежели хардкодить реплику в коде сценария.
            premise = bot.get_engine().find_premise('кто я', bot, session, interlocutor)
            assert(premise is not None)

            if self.activation_counter == 1:
                # Сценарий запускается первый раз.
                bot.say(session, premise)
            elif self.activation_counter == 2:
                # Уведомим пользователя, что мы уже отвечали на такой вопрос.
                bot.say(session, premise)
                sx = ['По-моему, я уже отвечала на этот вопрос...',
                      'Ты, кажется, уже спрашивал это',
                      'Вроде бы ты уже задавал этот вопрос',
                      'Я на такой вопрос уже давала ответ']
                bot.say(session, self.choice(bot, session, text_utils, sx))
            else:
                # Сценарий уже запускался ранее 2 раза.
                bot.say(session, premise)
                sx = ['Почему ты спрашиваешь снова и снова?',
                      'Почему ты уже в который раз задаешь этот вопрос?']
                bot.say(session, self.choice(bot, session, text_utils, sx))
        else:
            # Последующие шаги.
            if self.are_similar(bot, text_utils, interpreted_phrase.raw_phrase, 'дед пихто'):
                bot.say(session, 'Интересная у вас фамилия, дедушка!')
            elif self.are_similar(bot, text_utils, interpreted_phrase.raw_phrase, 'конь в пальто'):
                sx = ['С лошадьми мне еще не доводилось беседовать',
                      'С говорящим конем я еще не общалась']
                bot.say(session, self.choice(bot, session, text_utils, sx))
            else:
                sx = ['не хочу говорить', 'не хочу отвечать на этот вопрос',
                      'я не буду отвечать', 'не твоего ума дело', 'не скажу',
                      'давай я не буду отвечать на этот вопрос',
                      'не суй нос не в свои дела',
                      'это секрет',
                      'это секретная информация',
                      'это военная тайна',
                      'я не должен говорить тебе это',
                      'я не имею права рассказывать об этом',
                      'тебе лучше не знать об этом',
                      'я воздержусь от ответа',
                      'мне стыдно говорить об этом',
                      'я предпочту не отвечать на этот вопрос',
                      'давай об этом позже поговорим',
                      'давайте вернемся к этому в следующий раз',
                      'я не имею права рассказывать об этом',
                      'мне запрещено раскрывать эту информацию',
                      'иди в жопу с такими вопросами',
                      'тебе не следует задавать подобные вопросы'
                      ]
                if self.is_similar_any(bot, text_utils, interpreted_phrase.raw_phrase, sx):
                    sx = ['Хорошо, обсудим это в другой раз',
                          'Ладно, поговорим об этом в следующий раз',
                          'Как нибудь в другой раз спрошу еще раз',
                          'Ладно, не буду настаивать']
                    bot.say(session, self.choice(bot, session, text_utils, sx) )
                    bot.get_engine().exit_scenario(bot, session, interlocutor, interpreted_phrase)
                    return

                sx = ['ой',
                     'ну и вопросики у тебя',
                     'какая ты любопытная',
                     'ты очень любознательная',
                     'любопытной варваре на базаре нос оторвали',
                     'ох, серьезный вопрос']
                if self.is_similar_any(bot, text_utils, interpreted_phrase.raw_phrase, sx):
                    sx = ['Я просто хочу понимать, с кем разговариваю',
                          'Хочу узнать, чем ты занимаешься',
                          'Мне хотелось бы узнать, чем ты занимаешься',
                          'Хочу познакомиться с тобой',
                          'Хочу узнать о тебе']
                    bot.say(session, self.choice(bot, session, text_utils, sx) )

                # TODO тут надо сделать обработку прочих реплик пользователя...

            # Если мы не обработали реплику собеседника, то используем общие правила
            res = bot.get_engine().apply_insteadof_rule(bot.get_scripting().get_insteadof_rules(),
                                                        bot.get_scripting().get_story_rules(),
                                                        bot, session, interlocutor, interpreted_phrase)
            if not res.insteadof_applied:
                # Реплику собеседника обработать вообще не удалось
                pass

        # Если бот еще не знает, кто его собеседник - спросим.
        premise = bot.get_engine().find_premise('кто ты', bot, session, interlocutor)
        if premise is None:
            self.who_are_you_counter += 1
            if self.who_are_you_counter == 1:
                bot.say(session, 'Скажи, пожалуйста, кто ты?')
            elif self.who_are_you_counter == 2:
                bot.say(session, 'А все-таки, кто же ты?')
            elif self.who_are_you_counter == 3:
                bot.say(session, 'Не теряю надежду узнать, кто ты?')
            else:
                # сейчас просто закрываем сценарий...
                bot.say(session, 'Жаль, что ты не хочешь рассказать, кто ты')
                bot.get_engine().exit_scenario(bot, session, interlocutor, interpreted_phrase)

        else:
            # Так как боту известно, кто собеседник, то выходим из сценария
            if self.activation_counter == 1:
                sx = ['Теперь мы знаем, кто из нас кто',
                      'Вот и хорошо, что мы узнали кое-что друг про друга',
                      'Отлично, мы узнали друг про друга кое-что']
                bot.say(session, self.choice(bot, session, text_utils, sx))

            bot.get_engine().exit_scenario(bot, session, interlocutor, interpreted_phrase)

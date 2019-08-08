# coding: utf-8

import random

from ruchatbot.bot.running_form_status import RunningFormStatus


class ActorBase(object):
    def __init__(self, actor_keyword):
        self.actor_keyword = actor_keyword

    @staticmethod
    def from_yaml(yaml_node):
        actor_keyword = list(yaml_node.keys())[0] if isinstance(yaml_node, dict) else yaml_node
        if actor_keyword == 'say':
            return ActorSay.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'answer':
            return ActorAnswer.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'callback':
            return ActorCallback.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'form':
            return ActorForm.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'scenario':
            return ActorScenario.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'nothing':
            return ActorNothing()
        else:
            raise NotImplementedError(actor_keyword)

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        raise NotImplementedError()


class ActorNothing(ActorBase):
    def __init__(self):
        super(ActorNothing, self).__init__('nothing')

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        return False


class ActorSay(ActorBase):
    def __init__(self):
        super(ActorSay, self).__init__('say')
        self.phrases = []
        self.exhausted_phrases = []
        self.known_answer_policy = 'utter'

    @staticmethod
    def from_yaml(yaml_node):
        actor = ActorSay()

        # TODO: сделать расширенную диагностику ошибок описания!!!

        # Надо понять, тут расширенная форма описания актора или просто список реплик, возможно
        # из одного элемента.
        if isinstance(yaml_node, dict):
            # Расширенный формат.
            for inner_keyword in yaml_node.keys():
                if 'phrases' == inner_keyword:
                    for utterance in yaml_node['phrases']:
                        actor.phrases.append(utterance)
                elif 'exhausted' == inner_keyword:
                    for utterance in yaml_node['exhausted']:
                        actor.exhausted_phrases.append(utterance)
                elif 'known_answer' == inner_keyword:
                    actor.known_answer_policy = yaml_node[inner_keyword]
                    # TODO - проверить значение флага: 'skip' | 'utter'
                else:
                    raise NotImplementedError()

        elif isinstance(yaml_node, list):
            for utterance in yaml_node:
                if isinstance(utterance, str):
                    actor.phrases.append(utterance)
                else:
                    raise SyntaxError()
        elif isinstance(yaml_node, str):
            actor.phrases.append(yaml_node)

        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        # Сначала попробуем убрать из списка те реплики, которые мы уже произносили.
        new_utterances = []
        for utterance in self.phrases:
            if session.count_bot_phrase(utterance) == 0:
                if self.known_answer_policy == 'skip' and utterance[-1] == '?':
                    # Проверим, что бот еще не знает ответ на этот вопрос:
                    if bot.does_bot_know_answer(utterance, session, interlocutor):
                        continue

                new_utterances.append(utterance)

        uttered = False
        if len(new_utterances) > 0:
            # Выбираем одну из оставшихся фраз.
            if len(new_utterances) == 1:
                bot.say(session, new_utterances[0])
            else:
                bot.say(session, random.choice(new_utterances))
            uttered = True
        else:
            # Все фразы бот уже произнес
            # Если задан список фраз на случай исчерпания (типа "не знаю больше ничего про кошек"),
            # то выдадим одну из них.
            if len(self.exhausted_phrases) == 1:
                bot.say(session, self.exhausted_phrases[0])
                uttered = True
            elif len(self.exhausted_phrases) > 1:
                bot.say(session, random.choice(self.exhausted_phrases))
                uttered = True
            else:
                if self.known_answer_policy == 'skip':
                    pass
                else:
                    # Начиная с этого момента данное правило будет повторно выдавать
                    # одну из фраз.
                    bot.say(session, random.choice(self.phrases))
                    uttered = True

        return uttered


class ActorAnswer(ActorBase):
    """
    В качестве реплики бота ищется факт в базе фактов для указанного в правиле вопроса.
    Это позволяет, к примеру, не зашивать в правило имя бота и не размазывать эту информацию
    по множеству мест, а оставить факт "Меня зовут Вика" только в базе фактов, а в правиле
    заставить бота найти релевантный факт для вопроса "Как меня зовут?"
    """
    def __init__(self):
        super(ActorAnswer, self).__init__('answer')
        self.question = None
        self.output = None

    @staticmethod
    def from_yaml(yaml_node):
        actor = ActorAnswer()

        # TODO: сделать расширенную диагностику ошибок описания!!!

        # Надо понять, тут расширенная форма описания актора или просто список реплик, возможно
        # из одного элемента.
        if isinstance(yaml_node, dict):
            # Расширенный формат.
            for inner_keyword in yaml_node.keys():
                if 'question' == inner_keyword:
                    actor.question = yaml_node['question']
                elif 'output' == inner_keyword:
                    actor.output = yaml_node[inner_keyword]  # TODO: проверять что значение 'premise'
                else:
                    raise NotImplementedError()
        elif isinstance(yaml_node, str):
            actor.question = yaml_node
        else:
            raise NotImplementedError()

        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        if self.output and self.output == 'premise':
            premise = bot.get_engine().find_premise(self.question, bot, session, interlocutor)
            if premise:
                bot.say(session, premise)
        else:
            bot.push_phrase(interlocutor, self.question, True)

        return True


class ActorCallback(ActorBase):
    def __init__(self):
        super(ActorCallback, self).__init__('callback')
        self.event_name = None

    @staticmethod
    def from_yaml(yaml_node):
        actor = ActorCallback()
        assert(isinstance(yaml_node, str))
        actor.event_name = yaml_node
        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        # Если выполнялась вербальная форма и в ней заполнены поля (слоты), то
        # надо передать значения этих полей в обработчик.
        verb_form_fields = dict()
        status = session.get_status()
        if status:
            if isinstance(status, RunningFormStatus):
                verb_form_fields.update(status.fields)

        resp = bot.invoke_callback(self.event_name, session, interlocutor, interpreted_phrase, verb_form_fields)
        if resp:
            bot.say(session, resp)
        return True


class ActorForm(ActorBase):
    def __init__(self):
        super(ActorForm, self).__init__('form')
        self.form_name = None

    @staticmethod
    def from_yaml(yaml_node):
        actor = ActorForm()
        assert(isinstance(yaml_node, str))
        actor.form_name = str(yaml_node)
        # TODO остальные свойства формы
        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        bot.run_form(self, session, interlocutor, interpreted_phrase)
        return True


class ActorScenario(ActorBase):
    def __init__(self):
        super(ActorScenario, self).__init__('scenario')
        self.scenario_name = None

    @staticmethod
    def from_yaml(yaml_node):
        actor = ActorScenario()
        assert(isinstance(yaml_node, str))
        actor.scenario_name = str(yaml_node)
        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        bot.run_scenario(self, session, interlocutor, interpreted_phrase)
        return True

# coding: utf-8

import random

from ruchatbot.bot.running_form_status import RunningFormStatus
from ruchatbot.bot.interpreted_phrase import InterpretedPhrase
from ruchatbot.utils.constant_replacer import replace_constant


class ActorBase(object):
    def __init__(self, actor_keyword):
        self.actor_keyword = actor_keyword

    @staticmethod
    def from_yaml(yaml_node, constants, text_utils):
        actor_keyword = list(yaml_node.keys())[0] if isinstance(yaml_node, dict) else yaml_node
        if actor_keyword == 'say':
            return ActorSay.from_yaml(yaml_node[actor_keyword], constants, text_utils)
        elif actor_keyword == 'answer':
            return ActorAnswer.from_yaml(yaml_node[actor_keyword], constants, text_utils)
        elif actor_keyword == 'callback':
            return ActorCallback.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'form':
            return ActorForm.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'scenario':
            return ActorScenario.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'generate':
            return ActorGenerate.from_yaml(yaml_node[actor_keyword], constants, text_utils)
        elif actor_keyword == 'nothing':
            return ActorNothing()
        elif actor_keyword == 'state':
            return ActorState.from_yaml(yaml_node[actor_keyword])
        elif actor_keyword == 'steps':
            return ActorSteps.from_yaml(yaml_node[actor_keyword], constants, text_utils)
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
    def from_yaml(yaml_node, constants, text_utils):
        actor = ActorSay()

        # TODO: сделать расширенную диагностику ошибок описания!!!

        # Надо понять, тут расширенная форма описания актора или просто список реплик, возможно
        # из одного элемента.
        if isinstance(yaml_node, dict):
            # Расширенный формат.
            for inner_keyword in yaml_node.keys():
                if 'phrases' == inner_keyword:
                    for utterance in yaml_node['phrases']:
                        actor.phrases.append(replace_constant(utterance, constants, text_utils))
                elif 'exhausted' == inner_keyword:
                    for utterance in yaml_node['exhausted']:
                        actor.exhausted_phrases.append(replace_constant(utterance, constants, text_utils))
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
    def from_yaml(yaml_node, constants, text_utils):
        actor = ActorAnswer()

        # TODO: сделать расширенную диагностику ошибок описания!!!

        # Надо понять, тут расширенная форма описания актора или просто список реплик, возможно
        # из одного элемента.
        if isinstance(yaml_node, dict):
            # Расширенный формат.
            for inner_keyword in yaml_node.keys():
                if 'question' == inner_keyword:
                    actor.question = replace_constant(yaml_node['question'], constants, text_utils)
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
        self.mode = 'replace'

    @staticmethod
    def from_yaml(yaml_node):
        actor = ActorScenario()
        if isinstance(yaml_node, str):
            actor.scenario_name = str(yaml_node)
        elif isinstance(yaml_node, dict):
            actor.scenario_name = yaml_node['name']
            if 'mode' in yaml_node:
                actor.mode = yaml_node['mode']
                assert actor.mode in ('replace', 'call')
        else:
            raise NotImplementedError()

        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        bot.run_scenario(self, session, interlocutor, interpreted_phrase)
        return True


class ActorGenerate(ActorBase):
    """Генерация реплики по заданому шаблону и вывод результата от имени бота."""
    def __init__(self):
        super(ActorGenerate, self).__init__('generate')
        self.templates = []
        self.wordbag_questions = []
        self.wordbag_words = []

    @staticmethod
    def from_yaml(yaml_node, constants, text_utils):
        actor = ActorGenerate()

        if isinstance(yaml_node, dict):
            # Расширенный формат.
            for inner_keyword in yaml_node.keys():
                if 'templates' == inner_keyword:
                    for template in yaml_node['templates']:
                        actor.templates.append(replace_constant(template, constants, text_utils))
                elif 'template' == inner_keyword:
                    actor.templates.append(replace_constant(yaml_node[inner_keyword], constants, text_utils))
                elif 'wordbag_question' == inner_keyword:
                    actor.wordbag_questions.append(replace_constant(yaml_node[inner_keyword], constants, text_utils))
                elif 'wordbag_word' == inner_keyword:
                    actor.wordbag_words.append(replace_constant(yaml_node[inner_keyword], constants, text_utils))
                else:
                    raise NotImplementedError()
        elif isinstance(yaml_node, str):
            raise NotImplementedError()

        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        uttered = False

        wordbag = []
        for question in self.wordbag_questions:
            if bot.get_engine().does_bot_know_answer(question, bot, session, interlocutor):
                interpreted_phrase2 = InterpretedPhrase(question)
                answers = bot.get_engine().build_answers(session, bot, interlocutor, interpreted_phrase2)
                for answer in answers:
                    tokens = bot.get_engine().get_text_utils().tokenize(answer)
                    wordbag.extend((word, 1.0) for word in tokens)

        wordbag.extend((word, 1.0) for word in self.wordbag_words)

        if len(wordbag) > 0:
            replicas = []
            for template_str in self.templates:
                replicas1 = bot.get_engine().replica_grammar.generate_by_terms(template_str,
                                                                               wordbag,
                                                                               bot.get_engine().get_text_utils().known_words,
                                                                               use_assocs=False)
                replicas.extend(replicas1)

            if len(replicas) > 0:

                # Выбираем одну рандомную реплику среди сгенерированных.
                # TODO: взвешивать через модель уместности по контексту
                bot.say(session, bot.get_engine().select_relevant_replica(replicas, session, interlocutor))
                uttered = True

        return uttered


class ActorState(ActorBase):
    def __init__(self):
        super(ActorState, self).__init__('state')
        self.slot_name = None
        self.slot_value = None

    @staticmethod
    def from_yaml(yaml_node):
        actor = ActorState()
        actor.slot_name = yaml_node['slot']
        actor.slot_value = yaml_node['value']
        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        session.set_slot(self.slot_name, self.slot_value)
        return False


class ActorSteps(ActorBase):
    def __init__(self):
        super(ActorSteps, self).__init__('steps')
        self.steps = []

    @staticmethod
    def from_yaml(yaml_node, constants, text_utils):
        actor = ActorSteps()
        for y in yaml_node:
            a = ActorBase.from_yaml(y['step'], constants, text_utils)
            actor.steps.append(a)
        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase):
        uttered = False
        for a in self.steps:
            uttered |= a.do_action(bot, session, interlocutor, interpreted_phrase)

        return uttered

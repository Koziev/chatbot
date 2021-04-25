# coding: utf-8

import random
import logging

from ruchatbot.bot.running_form_status import RunningFormStatus
from ruchatbot.bot.interpreted_phrase import InterpretedPhrase
from ruchatbot.utils.constant_replacer import replace_constant
from ruchatbot.bot.saying_phrase import SayingPhrase, substitute_bound_variables
from ruchatbot.bot.rule_condition_matching import RuleConditionMatching
from ruchatbot.bot.phrase_token import PhraseToken


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
        elif actor_keyword == 'query':
            return ActorQueryDb.from_yaml(yaml_node[actor_keyword], constants, text_utils)
        else:
            raise NotImplementedError(actor_keyword)

    def do_action(self, bot, session, interlocutor, interpreted_phrase, text_utils):
        raise NotImplementedError()


class ActorNothing(ActorBase):
    def __init__(self):
        super(ActorNothing, self).__init__('nothing')

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
        return False


class ActorSay(ActorBase):
    def __init__(self):
        super(ActorSay, self).__init__('say')
        self.phrases = []  # list of SayingPhrase
        self.exhausted_phrases = []
        self.known_answer_policy = 'utter'
        self.np_sources = dict()
        self.on_repeat = []
        self.on_repeat_again = []

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
                        s = replace_constant(utterance, constants, text_utils)
                        actor.phrases.append(SayingPhrase(s))
                elif 'exhausted' == inner_keyword:
                    for utterance in yaml_node['exhausted']:
                        s = replace_constant(utterance, constants, text_utils)
                        actor.exhausted_phrases.append(SayingPhrase(s))
                elif 'on_repeat' == inner_keyword:
                    for utterance in yaml_node['on_repeat']:
                        s = replace_constant(utterance, constants, text_utils)
                        actor.on_repeat.append(SayingPhrase(s))
                elif 'on_repeat_again' == inner_keyword:
                    for utterance in yaml_node['on_repeat_again']:
                        s = replace_constant(utterance, constants, text_utils)
                        actor.on_repeat_again.append(SayingPhrase(s))
                elif 'known_answer' == inner_keyword:
                    actor.known_answer_policy = yaml_node[inner_keyword]
                    # TODO - проверить значение флага: 'skip' | 'utter'
                elif 'NP1' == inner_keyword:
                    actor.np_sources['NP1'] = yaml_node[inner_keyword]
                else:
                    raise NotImplementedError()

        elif isinstance(yaml_node, list):
            for utterance in yaml_node:
                if isinstance(utterance, str):
                    s = replace_constant(utterance, constants, text_utils)
                    actor.phrases.append(SayingPhrase(s))
                else:
                    raise SyntaxError()
        elif isinstance(yaml_node, str):
            s = replace_constant(yaml_node, constants, text_utils)
            actor.phrases.append(SayingPhrase(s))
        else:
            raise NotImplementedError()

        return actor

    def prepare4saying(self, phrase, condition_matching_results, text_utils):
        return substitute_bound_variables(phrase, condition_matching_results, text_utils)

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
        if self.np_sources:
            if condition_matching_results is None:
                condition_matching_results = RuleConditionMatching.create(True)

            for np, question in self.np_sources.items():
                    if bot.get_engine().does_bot_know_answer(question, bot, session, interlocutor):
                        interpreted_phrase2 = InterpretedPhrase(question)
                        answers = bot.get_engine().build_answers(session, bot, interlocutor, interpreted_phrase2)
                        if answers:
                            answer = answers[0]
                            tokens = text_utils.tokenize(answer)
                            tagsets = list(text_utils.postagger.tag(tokens))
                            lemmas = text_utils.lemmatizer.lemmatize(tagsets)

                            phrase_tokens = []
                            for word_index, (token, tagset, lemma) in enumerate(zip(tokens, tagsets, lemmas)):
                                t = PhraseToken()
                                t.word = token
                                t.norm_word = token.lower()
                                t.lemma = lemma[2]
                                t.tagset = tagset[1]
                                t.word_index = word_index
                                phrase_tokens.append(t)

                            condition_matching_results.add_group(np, tokens, phrase_tokens)
                        else:
                            return None

        session.actor_say_hit(id(self))
        if session.get_actor_say_hits(id(self)) > 1:
            # Выдадим реплику из отдельного списка, так как в текущей сессии это повторный вопрос
            new_utterances = []
            if self.on_repeat_again and session.get_actor_say_hits(id(self)) > 2:
                for utterance0 in self.on_repeat_again:
                    utterance = self.prepare4saying(utterance0, condition_matching_results, text_utils)
                    new_utterances.append(utterance)

            if len(new_utterances) == 0 and self.on_repeat:
                new_utterances = []
                for utterance0 in self.on_repeat:
                    utterance = self.prepare4saying(utterance0, condition_matching_results, text_utils)
                    new_utterances.append(utterance)

            if new_utterances:
                bot.say(session, random.choice(new_utterances))
                return True

        # Сначала попробуем убрать из списка те реплики, которые мы уже произносили.
        new_utterances = []
        for utterance0 in self.phrases:
            utterance = self.prepare4saying(utterance0, condition_matching_results, text_utils)

            if '$' in utterance:
                # Не удалось подставить значение в один из $-слотов, значит
                # надо исключить фразу.
                continue

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

            new_utterances = []
            for utterance0 in self.exhausted_phrases:
                utterance = self.prepare4saying(utterance0, condition_matching_results, text_utils)

                if '$' in utterance:
                    # Не удалось подставить значение в один из $-слотов, значит
                    # надо исключить фразу.
                    continue

                if session.count_bot_phrase(utterance) == 0:
                    if self.known_answer_policy == 'skip' and utterance[-1] == '?':
                        # Проверим, что бот еще не знает ответ на этот вопрос:
                        if bot.does_bot_know_answer(utterance, session, interlocutor):
                            continue

                    new_utterances.append(utterance)

            if new_utterances:
                bot.say(session, random.choice(new_utterances))
                uttered = True
            else:
                if self.known_answer_policy == 'skip':
                    pass
                else:
                    # Начиная с этого момента данное правило будет повторно выдавать
                    # одну из фраз.
                    #for src_phrase in sorted(self.phrases, key=lambda z: random.random()):
                    #    random_phrase = self.prepare4saying(src_phrase, condition_matching_results, text_utils)
                    #    if '$' not in random_phrase:
                    #        bot.say(session, random_phrase)
                    #        uttered = True
                    #        break
                    uttered = False

        return uttered


class ActorQueryDb(ActorBase):
    def __init__(self):
        super(ActorQueryDb, self).__init__('query')
        self.query_template = None  # SayingPhrase

    @staticmethod
    def from_yaml(yaml_node, constants, text_utils):
        actor = ActorQueryDb()
        s = replace_constant(yaml_node, constants, text_utils)
        actor.query_template = SayingPhrase(s)
        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
        query_str = substitute_bound_variables(self.query_template, condition_matching_results, text_utils)

        # TODO - запрос к движку inference, формирование результатов или выдача заглушки в случае отсутствия результата.
        query_results = ['Результаты выполнения запроса "{}"'.format(query_str)]
        for query_result in query_results:
            bot.say(session, query_result)
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

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
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

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
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

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
        bot.run_form(self, session, interlocutor, interpreted_phrase)
        return True


class ScenarioNotAvailableException(Exception):
    pass


class ActorScenario(ActorBase):
    def __init__(self):
        super(ActorScenario, self).__init__('scenario')
        self.scenario_name = None
        self.mode = 'replace'  # по умолчанию новый сценарий убирает все текущие сценарии (включая отложенные на стеке)

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

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
        if bot.scenarios_enabled:
            bot.run_scenario(self, session, interlocutor, interpreted_phrase)
            return True
        else:
            logging.getLogger('ActorScenario').warning('Scenarios are not enabled, can not run scenario "%s"', self.scenario_name)
            raise ScenarioNotAvailableException()


class ActorGenerate(ActorBase):
    """ Генерация реплики по заданому шаблону и вывод результата от имени бота. """
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
            actor.templates.append(replace_constant(yaml_node, constants, text_utils))
        elif isinstance(yaml_node, list):
            for s in yaml_node:
                actor.templates.append(replace_constant(s, constants, text_utils))
        else:
            raise NotImplementedError()

        return actor

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
        uttered = False

        wordbag = []
        if self.wordbag_questions:
            for question in self.wordbag_questions:
                if bot.get_engine().does_bot_know_answer(question, bot, session, interlocutor):
                    interpreted_phrase2 = InterpretedPhrase(question)
                    answers = bot.get_engine().build_answers(session, bot, interlocutor, interpreted_phrase2)
                    for answer in answers:
                        tokens = bot.get_engine().get_text_utils().tokenize(answer)
                        wordbag.extend((word, 1.0) for word in tokens)
        elif self.wordbag_words:
            wordbag.extend((word, 1.0) for word in self.wordbag_words)
        else:
            wordbag.extend((word, 1.0) for word in text_utils.tokenize(interpreted_phrase.raw_phrase))

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

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
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

    def do_action(self, bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils):
        uttered = False
        for a in self.steps:
            uttered |= a.do_action(bot, session, interlocutor, interpreted_phrase, condition_matching_results, text_utils)

        return uttered

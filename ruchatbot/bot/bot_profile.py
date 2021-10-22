"""
Конфигурация бота - флаги, настроечные константы.

13-04-2021 добавлен параметр "scenarios_enabled" для (раз)блокировки сценариев
"""

import json


class BotProfile(object):
    def __init__(self, bot_id=''):
        self.bot_id = bot_id
        self.profile = None
        self.premises_path = None
        self.faq_path = None
        self.smalltalk_generative_rules = None
        self.constants = dict()

    def get_id(self):
        return self.bot_id

    def _replace(self, path_str, data_dir, models_dir):
        return path_str.replace('$DATA', data_dir).replace('$MODELS', models_dir)

    def load(self, profile_path, data_dir, models_dir):
        with open(profile_path, 'r') as f:
            self.profile = json.load(f)

        # Файлы с фактами (База Знаний)
        self.premises_path = self._replace(self.profile['premises'], data_dir, models_dir)
        self.faq_path = self._replace(self.profile['faq'], data_dir, models_dir)
        self.rules_path = self._replace(self.profile['rules'], data_dir, models_dir)

        #self.smalltalk_generative_rules = self._replace(self.profile['smalltalk_generative_rules'],
        #                                                data_dir,
        #                                                models_dir)
        self.constants = self.profile['constants']

    @property
    def rules_enabled(self):
        return self.profile.get('rules_enabled', True)

    @property
    def smalltalk_enabled(self):
        return self.profile.get('smalltalk_enabled', True)

    @property
    def generative_smalltalk_enabled(self):
        return self.profile.get('generative_smalltalk_enabled', True)

    @property
    def force_question_answering(self):
        return self.profile.get('force_question_answering', False)

    @property
    def replica_after_answering(self):
        return self.profile.get('replica_after_answering', False)

    @property
    def scenarios_enabled(self):
        return self.profile.get('scenarios_enabled', True)

    @property
    def faq_enabled(self):
        return self.profile.get('faq_enabled', True)

    @property
    def confabulator_enabled(self):
        return self.profile.get('confabulator_enabled', True)

    @property
    def opposite_fact_comment_proba(self):
        return self.profile.get('opposite_fact_comment_proba', 0.2)

    @property
    def already_known_fact_comment_proba(self):
        return self.profile.get('already_known_fact_comment_proba', 0.2)

    @property
    def max_contradiction_comments(self):
        return self.profile.get('max_contradiction_comments', 2)

    # Политика формирования ответов в ответ на вопросы к боту ("как тебя зовут?")
    PERSONAL_QUESTIONS_ANSWERING__GENERAL = 'general'  # используется общий пайплайн с генерацией ответа
    PERSONAL_QUESTIONS_ANSWERING__PREMISE = 'premise'  # выдавать текст подобранной предпосылки в качестве ответа
    PERSONAL_QUESTIONS_ANSWERING__RANDOM  = 'random'   # рандомный выбор между двумя предыдущими способами

    @property
    def personal_question_answering_policy(self):
        return self.profile.get('personal_questions_answering_policy', BotProfile.PERSONAL_QUESTIONS_ANSWERING__GENERAL)

import json


class BotProfile(object):
    def __init__(self):
        self.profile = None
        self.premises_path = None
        self.faq_path = None
        self.smalltalk_generative_rules = None
        self.constants = dict()

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
        return self.profile.get('replica_after_answering', True)

    @property
    def premise_is_answer(self):
        return self.profile.get('premise_is_answer', False)

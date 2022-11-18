
class RunningDialogStatus(object):
    def __init__(self, priority):
        self.priority = priority

    def get_priority(self):
        return self.priority

    def get_greedy_rules(self):
        return None

    #def get_smalltalk_rules(self):
        return None

    #def get_story_rules(self):
    #    return None

    def get_name(self):
        raise NotImplementedError()

    def get_remaining_chitchat_questions_per_step(self):
        return 0

    def get_current_step_name(self):
        return ''

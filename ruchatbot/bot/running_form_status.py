# -*- coding: utf-8 -*-

from ruchatbot.bot.running_dialog_status import RunningDialogStatus


class RunningFormStatus(RunningDialogStatus):
    def __init__(self, form, interpreted_phrase, filled_fields, current_field):
        super(RunningFormStatus, self).__init__(100)
        self.form = form
        self.phrases = [interpreted_phrase]
        self.fields = filled_fields
        self.current_field = current_field

    def set_current_field(self, field):
        self.current_field = field

    def get_insteadof_rules(self):
        return self.form.insteadof_rules

    def get_smalltalk_rules(self):
        return self.form.smalltalk_rules

    def get_name(self):
        return self.form.get_name()

from ruchatbot.bot.running_dialog_status import RunningDialogStatus


class RunningScenario(RunningDialogStatus):
    def __init__(self, scenario, current_step_index):
        super(RunningScenario, self).__init__(scenario.get_priority())
        self.scenario = scenario
        self.current_step_index = current_step_index
        self.passed_steps = set()

    def get_insteadof_rules(self):
        return self.scenario.insteadof_rules

    def get_smalltalk_rules(self):
        return self.scenario.smalltalk_rules

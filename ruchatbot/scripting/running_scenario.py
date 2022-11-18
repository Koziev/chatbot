import collections

from ruchatbot.scripting.running_dialog_status import RunningDialogStatus


class RunningScenario(RunningDialogStatus):
    """
    Переменные состояния исполняющегося сценария.
    Экземпляр создается при запуске сценария и удаляется после завршения сценария,
    поэтому накопленная в этой структуре информация о выполнении шагов не будет
    влиять на новый запуск сценария.
    """
    def __init__(self, scenario, current_step_index):
        super(RunningScenario, self).__init__(scenario.get_priority())
        self.scenario = scenario
        self.current_step_index = current_step_index
        self.passed_steps = set()
        self.istep_2_chitchat_questions = collections.defaultdict(lambda: 0)

    def get_greedy_rules(self):
        return self.scenario.greedy_rules

    #def get_smalltalk_rules(self):
    #    return self.scenario.smalltalk_rules

    #def get_story_rules(self):
    #    return self.scenario.story_rules

    def get_name(self):
        return self.scenario.get_name()

    def get_remaining_chitchat_questions_per_step(self):
        """Вернет оставшееся число вопросов, сгенерированных читчатом, которые еще можно выдать
        для текущего шага в диалоге"""
        x = self.scenario.get_chitchat_questions_per_step_rate()
        if x > 0:
            y = self.istep_2_chitchat_questions[self.current_step_index]
            self.istep_2_chitchat_questions[self.current_step_index] = y + 1
            remainder = x - y
            return remainder

        return 0

    def get_current_step_name(self):
        if self.current_step_index == -1:
            return '-1'
        else:
            return self.scenario.get_step_name(self.current_step_index)

import logging
import os

from ruchatbot.bot.rugpt_base import RugptBase
from ruchatbot.bot.base_utterance_interpreter2 import BaseUtteranceInterpreter2


class RugptInterpreter(RugptBase, BaseUtteranceInterpreter2):
    def __init__(self):
        BaseUtteranceInterpreter2.__init__(self)
        RugptBase.__init__(self)

    def load(self, models_dir):
        BaseUtteranceInterpreter2.load(self, models_dir)
        p = os.path.join(models_dir, 'rugpt_interpreter')
        logging.debug('Start loading interpreter model from "%s"', p)
        self.load_from_path(p)

    def generate_output(self, context, num_return_sequences):
        prompt_text = '<s>' + context + ' #'
        return self.generate_output_from_prompt(prompt_text, num_return_sequences)

    def interpret(self, phrases, num_return_sequences):
        prompt_text = '<s>' + '\n'.join(('- ' + f) for f in phrases) + ' #'
        outputs = self.generate_output_from_prompt(prompt_text, num_return_sequences)
        return outputs


if __name__ == '__main__':
    model = RugptInterpreter()
    model.load('../../../tmp')

    lines = []
    while True:
        q = input(':> ').strip()
        if q:
            lines.append(q)
        else:
            ctx = '\n'.join(('- '+f) for f in lines)
            ox = model.generate_output(ctx, num_return_sequences=10)
            for i, o in enumerate(ox, start=1):
                print('[{}] {}'.format(i, o))
            print('='*60)
            lines = []

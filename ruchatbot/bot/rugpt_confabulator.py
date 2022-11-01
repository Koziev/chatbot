import logging.handlers
import os

from ruchatbot.bot.rugpt_base import RugptBase


class RugptConfabulator(RugptBase):
    def __init__(self):
        super(RugptConfabulator, self).__init__()

    def load(self, models_dir):
        p = os.path.join(models_dir, 'rugpt_premise4question')
        logging.debug('Start loading confabulation model from "%s"', p)
        self.load_from_path(p)

    def generate_confabulations(self, question, num_return_sequences):
        prompt_text = '<s>- ' + question + '\n-'
        raw_output = self.generate_output_from_prompt(prompt_text, num_return_sequences)
        return [s for s in raw_output if not s.endswith('?')]


if __name__ == '__main__':
    model = RugptConfabulator()
    model.load('../../../tmp')

    while True:
        q = input(':> ').strip()
        if q:
            ox = model.generate_confabulations(q, num_return_sequences=20)
            for i, o in enumerate(ox, start=1):
                print('[{}] {}'.format(i, o))
            print('='*60)
        else:
            break


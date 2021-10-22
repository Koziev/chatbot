import logging.handlers
import os

from rugpt_base import RugptBase


class RugptConfabulator(RugptBase):
    def __init__(self):
        super(RugptConfabulator, self).__init__()

    def load(self, models_dir):
        p = os.path.join(models_dir, 'rugpt_premise4question')
        logging.debug('Start loading confabulator model from "%s"', p)
        self.load_from_path(p)

    def generate_output(self, context, num_return_sequences):
        prompt_text = '<s>' + context + ' # '
        return self.generate_output_from_prompt(prompt_text, num_return_sequences)


if __name__ == '__main__':
    model = RugptConfabulator()
    model.load('../../../tmp')

    while True:
        q = input(':> ').strip()
        if q:
            ox = model.generate_output(q, num_return_sequences=2)
            for i, o in enumerate(ox, start=1):
                print('[{}] {}'.format(i, o))
            print('='*60)
        else:
            break


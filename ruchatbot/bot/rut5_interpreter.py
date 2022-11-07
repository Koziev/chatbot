"""
Обертка для использования модели Incomplete Utterance Restoration на базе отфайнтбненной ruT5 в чатботе.
"""

import logging
import os

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ruchatbot.bot.base_utterance_interpreter2 import BaseUtteranceInterpreter2


class RuT5Interpreter(BaseUtteranceInterpreter2):
    def __init__(self, device=None):
        BaseUtteranceInterpreter2.__init__(self)
        if device is None:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = device

    def load(self, models_dir):
        BaseUtteranceInterpreter2.load(self, models_dir)
        p = os.path.join(models_dir, 't5_interpreter')
        logging.debug('Start loading T5 Interpreter model from "%s"', p)
        self.tokenizer = T5Tokenizer.from_pretrained(p)
        self.model = T5ForConditionalGeneration.from_pretrained(p)
        self.model.to(self.device)
        self.model.eval()

    def interpret(self, phrases, num_return_sequences):
        t5_input = '\n'.join(('- ' + f) for f in phrases)
        input_ids = self.tokenizer(t5_input, return_tensors='pt').input_ids.to(self.device)
        out_ids = self.model.generate(input_ids=input_ids,
                                      max_length=60,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                      do_sample=True,
                                      temperature=1.0,
                                      num_return_sequences=num_return_sequences,
                                      )

        outputs = set()
        for i in range(len(out_ids)):
            o = self.tokenizer.decode(out_ids[i][1:])
            o = o[:o.index('</s>')]
            outputs.add(o)

        return list(outputs)


if __name__ == '__main__':
    model = RuT5Interpreter()
    model.load(os.path.expanduser('~/polygon/chatbot/tmp'))

    lines = []
    while True:
        q = input(':> ').strip()
        if q:
            lines.append(q)
        else:
            ox = model.interpret(lines, num_return_sequences=10)
            for i, o in enumerate(ox, start=1):
                print('[{}] {}'.format(i, o))
            print('='*60)
            lines = []

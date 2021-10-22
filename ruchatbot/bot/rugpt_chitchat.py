"""
Модель для генерации реплик в чит-чате
Часть пайплайна чатбота https://github.com/Koziev/chatbot
"""

import logging.handlers

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class RugptChitChat:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.logger = logging.getLogger('RugptChitChat')
        self.temperature = 0.9
        self.beam_k = 10
        self.beam_p = 0.9
        self.repetition_penalty = 1.0

    def load(self, model_name_or_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)

    def generate_output(self, context, num_return_sequences=10):
        self.logger.debug('Generating chit-chat response with context=%s', context.replace('\n', ' | '))

        nspaces = context.count(' ')
        if nspaces == 0:
            # По однословным контекстам не будем генерировать отклик.
            return []

        prompt_text = context + '\n'
        stop_token = "</s>"
        length = 80

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=self.temperature,
            top_k=self.beam_k,
            top_p=self.beam_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=0
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = set()
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            #print("ruGPT2Large:".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            if stop_token in text:
                text = text[: text.find(stop_token)]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            if total_sequence.startswith('- '):
                total_sequence = total_sequence[1:]

            if '\n' in total_sequence:
                total_sequence = total_sequence[:total_sequence.index('\n')]

            total_sequence = total_sequence.strip()
            generated_sequences.add(total_sequence)

        self.logger.debug('Chit-chat generated %d responses: %s', len(generated_sequences), '; '.join(generated_sequences))
        return list(generated_sequences)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)

    chitchat = RugptChitChat()
    chitchat.load('/home/inkoziev/polygon/chatbot/tmp/rugpt_chitchat')

    context = []
    while True:
        q = input(':> ').strip()
        if q:
            context.append(q)
        else:
            if context:
                context = '\n'.join(context)
                px = chitchat.generate_output(context)
                for p in px:
                    print('{}'.format(p))
                print('')
                context = []

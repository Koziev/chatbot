"""
Модель для генерации реплик в чит-чате
Часть пайплайна чатбота https://github.com/Koziev/chatbot

08.10.2022 Вычисление перплексии модели на диалогах переделано на прогон батчем.
"""

import logging.handlers
import math

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import CrossEntropyLoss


def pad_tokens(tokens, max_len):
    l = len(tokens)
    return tokens + [0] * (max_len - l)


class RugptChitChat:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.logger = logging.getLogger('RugptChitChat')
        self.temperature = 1.0
        self.top_k = 30
        self.top_p = 0.9
        self.repetition_penalty = 1.2

    def load(self, model_name_or_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def generate_chitchat(self, context_replies, num_return_sequences):
        return self.generate_output(context_replies, num_return_sequences)

    def generate_output(self, lines, num_return_sequences=10):
        self.logger.debug('Generating chit-chat response with context=〚%s〛', ' | '.join(lines))

        prompt_text = '\n'.join(('- '+line) for line in lines) + '\n-'
        stop_token = "</s>"
        length = 80

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
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
            generated_sequence = generated_sequence.tolist()

            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            if stop_token in text:
                text = text[: text.find(stop_token)]

            total_sequence = text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            if total_sequence.startswith('- '):
                total_sequence = total_sequence[1:]

            if '\n' in total_sequence:
                total_sequence = total_sequence[:total_sequence.index('\n')]

            total_sequence = total_sequence.strip()
            generated_sequences.add(total_sequence)

        self.logger.debug('Chit-chat generated %d responses: %s', len(generated_sequences), '; '.join(generated_sequences))
        return list(generated_sequences)

    def score_dialogues(self, dialogues):
        """ Вычисляем перплексию множества диалогов одним батчем """
        scores = []

        encoded_texts = [self.tokenizer.encode('<s>' + '\n'.join(dialog) + '</s>') for dialog in dialogues]
        max_len = max(map(len, encoded_texts))
        padded_texts = [pad_tokens(tokens, max_len) for tokens in encoded_texts]
        input_ids = torch.tensor(padded_texts, dtype=torch.long, device=self.device)

        target_ids = input_ids.clone()
        for row, tokens in enumerate(encoded_texts):
            target_ids[row, len(tokens):] = -100   # хвосты из <pad>-ов не учитываем при расчете лоссов

        with torch.no_grad():
            o = self.model(input_ids)
            for irow in range(len(dialogues)):
                # Shift so that tokens < n predict n
                shift_logits = o.logits[irow, :-1, :].contiguous()
                shift_labels = target_ids[irow, 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                score = math.exp(-loss)
                scores.append(math.exp(-score))

        return scores


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)

    chitchat = RugptChitChat()
    #chitchat.load('/home/inkoziev/polygon/chatbot/tmp/rugpt_chitchat.1')
    chitchat.load('/home/inkoziev/polygon/chatbot/tmp/rugpt_npqa')

    context = []
    while True:
        q = input(':> ').strip()
        if q:
            context.append(q)
        else:
            if context:
                px = chitchat.generate_output(context)
                for p in px:
                    print('{}'.format(p))
                print('')
                context = []

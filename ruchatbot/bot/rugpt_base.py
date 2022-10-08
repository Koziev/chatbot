import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class RugptBase:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.beam_k = 20
        self.beam_p = 0.9

    def load_from_path(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def generate_output_from_prompt(self, prompt_text, num_return_sequences, temperature=1.0):
        repetition_penalty = 1.2
        stop_token = "</s>"
        length = 100

        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=self.beam_k,
            top_p=self.beam_p,
            repetition_penalty=repetition_penalty,
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

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            if stop_token in text:
                text = text[: text.find(stop_token)]

            total_sequence = text[len(prompt_text):].replace('<pad>', '').strip()
            if total_sequence:
                generated_sequences.add(total_sequence)

        return list(generated_sequences)

"""
Обертка для инференса модели диалогового entailment'а

15-03-2022 Добавлен метод predictN для обработки батча вариантов ответа
"""

import os
import json

import torch
import transformers

from ruchatbot.bot.rubert_entailment_detector import RubertEntailmentDetector


class EntailmentModel:
    def __init__(self, device):
        self.device = device
        self.config = None
        self.model = None
        self.pad_token_id = 0

    def load(self, models_dir, bert_model, bert_tokenizer):
        weights_path = os.path.join(models_dir, 'rubert_entailment.pt')
        config_path = os.path.join(models_dir, 'rubert_entailment.cfg')

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.max_len = self.config['max_len']

        self.model = RubertEntailmentDetector(device=self.device, **self.config)
        self.model.load_weights(weights_path)
        self.model.eval()
        self.model.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

    def pad_tokens(self, tokens):
        l = len(tokens)
        if l > self.max_len:
            return tokens[:self.max_len]
        elif l < self.max_len:
            return tokens + [self.pad_token_id] * (self.max_len - l)
        else:
            return tokens

    def predict1(self, dialog_context, response):
        assert(isinstance(dialog_context, str))
        assert(isinstance(response, str))
        tokens1 = self.bert_tokenizer.encode(dialog_context)
        tokens2 = self.bert_tokenizer.encode(response)

        z1 = torch.tensor(self.pad_tokens(tokens1)).reshape(1, self.max_len).to(self.device)
        z2 = torch.tensor(self.pad_tokens(tokens2)).reshape(1, self.max_len).to(self.device)

        with torch.no_grad():
            y = self.model(z1, z2).cpu()
            return y[0].item()

    def predictN(self, dialog_context, responses):
        assert(isinstance(dialog_context, str))

        # диалоговый контекст у всех вариантов реплик одинаковый!
        tokens1 = self.pad_tokens(self.bert_tokenizer.encode(dialog_context))

        matrix1 = []
        matrix2 = []
        for response in responses:
            matrix1.append(tokens1)
            tokens2 = self.pad_tokens(self.bert_tokenizer.encode(response))
            matrix2.append(tokens2)

        z1 = torch.tensor(matrix1).to(self.device)
        z2 = torch.tensor(matrix2).to(self.device)

        with torch.no_grad():
            y = self.model(z1, z2).cpu()
            if len(y.shape) == 0:
                return [y.item()]
            else:
                return [yy[0].item() for yy in y]


if __name__ == '__main__':
    model = EntailmentModel()
    model.load('../../../tmp')
    p = model.predict1('тебя как зовут?', 'меня зовут Вика')
    print(p)

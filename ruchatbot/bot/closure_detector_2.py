"""
Модель проверки полноты входного контекста для ответа на вопрос.
"""

import torch.utils.data
import torch
import torch.nn as nn
import torch.utils.data


class RubertClosureDetector0(nn.Module):
    def __init__(self, device, arch, max_len, sent_emb_size):
        super(RubertClosureDetector0, self).__init__()
        self.max_len = max_len
        self.arch = arch

        if self.arch == 1:
            #self.norm = torch.nn.BatchNorm1d(num_features=sent_emb_size)
            self.fc1 = nn.Linear(sent_emb_size, 20)
            self.fc2 = nn.Linear(20, 1)
        elif self.arch == 2:
            self.rnn = nn.LSTM(input_size=sent_emb_size, hidden_size=sent_emb_size, num_layers=1, batch_first=True, bidirectional=True)
            self.fc1 = nn.Linear(in_features=sent_emb_size*2, out_features=20)
            self.fc2 = nn.Linear(in_features=20, out_features=1)
        elif self.arch == 3:
            cnn_size = 100
            self.conv = nn.Conv1d(sent_emb_size, out_channels=cnn_size, kernel_size=3)
            self.fc1 = nn.Linear(in_features=cnn_size, out_features=1)
        else:
            raise NotImplementedError()

        self.device = device
        self.to(device)

    def save_weights(self, weights_path):
        # !!! Не сохраняем веса rubert, так как они не меняются при обучении и одна и та же rubert используется
        # несколькими моделями !!!
        state = dict((k, v) for (k, v) in self.state_dict().items() if not k.startswith('bert_model'))
        torch.save(state, weights_path)

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.eval()
        return

    def forward_0(self, x, bert_output):
        mask0 = (x != 0)
        mask = mask0.unsqueeze(2)  # чтобы исключить pad-токены из расчета лосса

        if self.arch == 1:
            #w = b.sum(dim=-2)
            #w = (b * mask).sum(dim=-2) / mask0.sum(dim=-1).unsqueeze(1)
            w = bert_output.pooler_output

            #z = self.norm(w)
            z = w

            merged = self.fc1(z)
            merged = torch.relu(merged)
            merged = self.fc2(merged)
            output = torch.sigmoid(merged)

        elif self.arch == 2:
            b = bert_output.last_hidden_state

            #out1, (hidden1, cell1) = self.rnn(b)
            #v1 = out1[:, -1, :]

            # исключаем pad-токены из обработки в LSTM
            pack = torch.nn.utils.rnn.pack_padded_sequence(b, lengths=mask0.sum(dim=-1).detach().cpu().numpy(), batch_first=True, enforce_sorted=False)
            out1, (hidden1, cell1) = self.rnn(pack)
            v1 = torch.hstack((hidden1[0, :, :], hidden1[1, :, :]))

            merged = self.fc1(v1)
            merged = torch.sigmoid(merged)
            merged = self.fc2(merged)
            output = torch.sigmoid(merged)

        elif self.arch == 3:
            z = (bert_output.last_hidden_state * mask).transpose(1, 2).contiguous()

            v = self.conv1(z)
            v = torch.relu(v).transpose(1, 2).contiguous()

            net, _ = torch.max(v, 1)
            net = self.fc1(net)
            output = torch.sigmoid(net)
        else:
            raise NotImplementedError()

        return output

    def pad_tokens(self, tokens):
        l = len(tokens)
        if l < self.max_len:
            return tokens + [0] * (self.max_len - l)
        elif l > self.max_len:
            return tokens[:self.max_len]
        else:
            return tokens


class RubertClosureDetector(RubertClosureDetector0):
    """Вариант с внутренним вызовом rubert"""
    def __init__(self, device, arch, max_len, sent_emb_size, **kwargs):
        super(RubertClosureDetector, self).__init__(device, arch, max_len, sent_emb_size)
        self.bert_tokenizer = None
        self.bert_model = None

    def forward(self, x):
        with torch.no_grad():
            bb = self.bert_model(x)

        return self.forward_0(x, bb)

    def calc_label(self, text):
        tokens = self.pad_tokens(self.bert_tokenizer.encode(text))
        z = torch.unsqueeze(torch.tensor(tokens), 0).to(self.device)
        y = self.forward(z)[0].item()
        return y

    def score_contexts(self, texts):
        if texts:
            tokenized_texts = [self.pad_tokens(self.bert_tokenizer.encode(text)) for text in texts]
            z = torch.tensor(tokenized_texts).to(self.device)
            ys = self.forward(z).detach().cpu().squeeze().tolist()
            return ys
        else:
            return []


class RubertClosureDetector_2(RubertClosureDetector0):
    """Вариант с внешним вызовом rubert"""
    def __init__(self, device, arch, max_len, sent_emb_size):
        super(RubertClosureDetector_2, self).__init__(device, arch, max_len, sent_emb_size)

    def forward(self, b):
        return self.forward_0(b)

    def calc_label(self, text):
        tokens = self.pad_tokens(self.bert_tokenizer.encode(text))
        z = torch.unsqueeze(torch.tensor(tokens), 0).to(self.device)

        with torch.no_grad():
            b = self.bert_model(z)[0]

        y = self.forward_0(b)[0].item()
        return y

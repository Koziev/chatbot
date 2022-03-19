"""
Модель для определения семантической близости двух текстов.
Используется rugpt (веса не меняются) и добавочные слои для классификатора.

13.03.2022 Скорректирован код обучения, чтобы веса rubert не участвовали в градиентном спуске
"""

import collections

import torch
import torch.nn as nn
import torch.utils.data


class RubertSynonymyDetector(nn.Module):
    def __init__(self, device, arch, max_len, sent_emb_size):
        super(RubertSynonymyDetector, self).__init__()
        self.max_len = max_len
        self.arch = arch

        if self.arch == 1:
            self.fc1 = nn.Linear(sent_emb_size*2, 20)
            self.fc2 = nn.Linear(20, 1)
        elif self.arch == 2:
            embedding_dim = sent_emb_size
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
            self.fc1 = nn.Linear(in_features=embedding_dim*4, out_features=20)
            self.fc2 = nn.Linear(in_features=20, out_features=1)
        elif self.arch == 3:
            cnn_size = 200
            self.conv1 = nn.Conv1d(sent_emb_size, out_channels=cnn_size, kernel_size=3)
            self.conv2 = nn.Conv1d(sent_emb_size, out_channels=cnn_size, kernel_size=3)
            self.fc1 = nn.Linear(in_features=cnn_size * 4, out_features=1)
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
        return

    def forward(self, x1, x2):
        with torch.no_grad():
            b1 = self.bert_model(x1)[0]
            b2 = self.bert_model(x2)[0]

        if self.arch == 1:
            w1 = b1.sum(dim=-2)
            w2 = b2.sum(dim=-2)

            z1 = self.norm(w1)
            z2 = self.norm(w2)

            #merged = torch.cat((z1, z2, torch.abs(z1 - z2)), dim=-1)
            #merged = torch.cat((z1, z2, torch.abs(z1 - z2), z1 * z2), dim=-1)
            merged = torch.cat((z1, z2), dim=-1)

            merged = self.fc1(merged)
            merged = torch.relu(merged)
            merged = self.fc2(merged)
            output = torch.sigmoid(merged)

        elif self.arch == 2:
            out1, (hidden1, cell1) = self.rnn(b1)
            v1 = out1[:, -1, :]

            out2, (hidden2, cell2) = self.rnn(b2)
            v2 = out2[:, -1, :]

            v_sub = torch.sub(v1, v2)
            v_mul = torch.mul(v1, v2)

            merged = torch.cat((v1, v2, v_sub, v_mul), dim=-1)

            merged = self.fc1(merged)
            merged = torch.sigmoid(merged)
            merged = self.fc2(merged)
            output = torch.sigmoid(merged)

        elif self.arch == 3:
            z1 = b1.transpose(1, 2).contiguous()
            z2 = b2.transpose(1, 2).contiguous()

            v1 = self.conv1(z1)
            v1 = torch.relu(v1).transpose(1, 2).contiguous()

            v2 = self.conv2(z2)
            v2 = torch.relu(v2).transpose(1, 2).contiguous()

            v_sub = torch.sub(v1, v2)
            v_mul = torch.mul(v1, v2)

            merged = torch.cat((v1, v2, v_sub, v_mul), dim=-1)

            net, _ = torch.max(merged, 1)
            net = self.fc1(net)
            output = torch.sigmoid(net)
        else:
            raise NotImplementedError()

        return output

    def pad_tokens(self, tokens):
        l = len(tokens)
        if l < self.max_len:
            return tokens + [0] * (self.max_len - l)
        else:
            return tokens

    def calc_synonymy1(self, text1, text2):
        tokens1 = self.pad_tokens(self.bert_tokenizer.encode(text1))
        tokens2 = self.pad_tokens(self.bert_tokenizer.encode(text2))

        z1 = torch.unsqueeze(torch.tensor(tokens1), 0).to(self.device)
        z2 = torch.unsqueeze(torch.tensor(tokens2), 0).to(self.device)

        y = self.forward(z1, z2)[0].item()
        return y

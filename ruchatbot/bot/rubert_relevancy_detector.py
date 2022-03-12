"""
Модель детектора PQ-релевантности с использованием претренированной модели BERT на PyTorch

Подготовка обучающего датасета - в preparation/prepare_relevancy_dataset.py
Тренировка - в trainers/train_rubert_pq_relevancy_detector.py

09.03.2022 Начальная реализация на базе кода детектора перефразировок
10.03.2022 Переделан код сохранения и загрузки модели, чтобы не дублировать веса rubert.
10.03.2022 Сделан фриз весов rubert при обучении
11.03.2022 Класс RubertRelevancyDetector выделен в отдельный модуль, чтобы унифицировать использование в диалоговом движке
"""

import torch.utils.data
import torch
import torch.nn as nn
import torch.utils.data


class RubertRelevancyDetector(nn.Module):
    def __init__(self, device, arch, max_len, sent_emb_size):
        super(RubertRelevancyDetector, self).__init__()
        self.max_len = max_len
        self.arch = arch

        if self.arch == 1:
            self.norm = torch.nn.BatchNorm1d(num_features=sent_emb_size)
            self.fc1 = nn.Linear(sent_emb_size*2, 20)
            self.fc2 = nn.Linear(20, 1)
        elif self.arch == 2:
            self.rnn1 = nn.LSTM(input_size=sent_emb_size, hidden_size=sent_emb_size, num_layers=1, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=sent_emb_size, hidden_size=sent_emb_size, num_layers=1, batch_first=True)
            self.fc1 = nn.Linear(in_features=sent_emb_size*4, out_features=20)
            self.fc2 = nn.Linear(in_features=20, out_features=1)
        elif self.arch == 3:
            cnn_size = 100
            self.conv1 = nn.Conv1d(sent_emb_size, out_channels=cnn_size, kernel_size=3)
            self.conv2 = nn.Conv1d(sent_emb_size, out_channels=cnn_size, kernel_size=3)
            self.fc1 = nn.Linear(in_features=cnn_size*4, out_features=1)
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
            out1, (hidden1, cell1) = self.rnn1(b1)
            v1 = out1[:, -1, :]

            out2, (hidden2, cell2) = self.rnn2(b2)
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

    def calc_relevancy1(self, premise, query, **kwargs):
        tokens1 = self.pad_tokens(self.bert_tokenizer.encode(premise))
        tokens2 = self.pad_tokens(self.bert_tokenizer.encode(query))

        z1 = torch.unsqueeze(torch.tensor(tokens1), 0).to(self.device)
        z2 = torch.unsqueeze(torch.tensor(tokens2), 0).to(self.device)

        y = self.forward(z1, z2)[0].item()
        return y

    def get_most_relevant(self, query, premises, text_utils, nb_results=1):
        query_t1 = self.pad_tokens(self.bert_tokenizer.encode(query))

        res = []

        batch_size = 100
        while premises:
            premises_batch = premises[:batch_size]
            premises = premises[batch_size:]
            premises_tx = [self.pad_tokens(self.bert_tokenizer.encode(premise)) for premise, _, _ in premises_batch]
            query_tx = [query_t1 for _ in range(len(premises_batch))]

            z1 = torch.tensor(premises_tx).to(self.device)
            z2 = torch.tensor(query_tx).to(self.device)

            y = self.forward(z1, z2).squeeze()
            delta = [(premise[0], yi.item()) for (premise, yi) in zip(premises_batch, y)]
            res.extend(delta)

        res = sorted(res, key=lambda z: -z[1])[:nb_results]
        return [x[0] for x in res], [x[1] for x in res]

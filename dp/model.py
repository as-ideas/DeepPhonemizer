from typing import List, Tuple

import torch
import torch.nn as nn
import math


from dp.text import Preprocessor


class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bnorm(x)
        x = x.transpose(1, 2)
        return x


class Aligner(torch.nn.Module):

    def __init__(self,
                 num_symbols_in: int,
                 num_symbols_out: int,
                 lstm_dim: int,
                 conv_dim: int) -> None:
        super().__init__()
        self.register_buffer('step', torch.tensor(1, dtype=torch.int))
        self.embedding = nn.Embedding(num_embeddings=num_symbols_in, embedding_dim=conv_dim)
        self.rnn = torch.nn.LSTM(conv_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols_out)

    def forward(self, x):
        if self.train:
            self.step += 1
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x

    def generate(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.lin(x)
        x = x.argmax(2)

        return x, None

    def get_step(self):
        return self.step.data.item()

    @classmethod
    def from_config(cls, config: dict) -> 'Aligner':
        preprocessor = Preprocessor.from_config(config)
        model = Aligner(
            num_symbols_in=preprocessor.text_tokenizer.vocab_size,
            num_symbols_out=preprocessor.phoneme_tokenizer.vocab_size,
            lstm_dim=config['model']['lstm_dim'],
            conv_dim=config['model']['conv_dim']
        )
        return model
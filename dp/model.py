from typing import List, Tuple

import torch
import torch.nn as nn
import math

from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

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


class AlignerLstm(torch.nn.Module):

    def __init__(self,
                 num_symbols_in: int,
                 num_symbols_out: int,
                 lstm_dim: int,
                 conv_dim: int) -> None:
        super().__init__()
        self.register_buffer('step', torch.tensor(1, dtype=torch.int))
        self.embedding = nn.Embedding(num_embeddings=num_symbols_in, embedding_dim=conv_dim)
        self.convs = nn.ModuleList([
            BatchNormConv(conv_dim, conv_dim, 5),
            BatchNormConv(conv_dim, conv_dim, 5),
            BatchNormConv(conv_dim, conv_dim, 5),
        ])
        self.rnn = torch.nn.LSTM(conv_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.rnn_2 = torch.nn.LSTM(2*conv_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols_out)

    def forward(self, x):
        if self.training:
            self.step += 1
        x = self.embedding(x)
        #for conv in self.convs:
        #    x = conv(x)
        x, _ = self.rnn(x)
        x, _ = self.rnn_2(x)
        x = self.lin(x)
        return x

    def generate(self, x):
        x = self.embedding(x)
        #for conv in self.convs:
        #    x = conv(x)
        x, _ = self.rnn(x)
        x, _ = self.rnn_2(x)
        x = self.lin(x)
        x_out = x.argmax(2)

        return x_out, x

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





class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):         # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class Aligner(nn.Module):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=1024,
                 layers=4,
                 dropout=0.1,
                 heads=1):
        super(Aligner, self).__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=heads,
                                                dim_feedforward=d_fft,
                                                dropout=dropout,
                                                activation='relu')
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layers,
                                          norm=encoder_norm)

        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

        self.register_buffer('step', torch.tensor(1, dtype=torch.int))

        self.src_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, x):         # shape: [N, T]

        if self.training:
            self.step += 1

        x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = self.make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x

    def generate(self,
                 x,           # shape: [N, T]
                ) -> Tuple[torch.tensor, torch.tensor]:

        """ Returns indices and logits """

        if self.training:
            self.step += 1

        x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = self.make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        x_out = x.argmax(2)

        return x_out, x

    def get_step(self):
        return self.step.data.item()

    @classmethod
    def from_config(cls, config: dict) -> 'Aligner':
        preprocessor = Preprocessor.from_config(config)
        return Aligner(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )

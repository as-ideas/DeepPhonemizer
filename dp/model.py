from typing import List

import torch
import torch.nn as nn
import math


# https://colab.research.google.com/drive/1g4ZFCGegOmD-xXL-Ggu7K5LVoJeXYJ75
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


class TransformerModel(nn.Module):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 decoder_start_index: int,
                 decoder_end_index: int,
                 d_model=512,
                 d_fft=1024,
                 encoder_layers=4,
                 decoder_layers=4,
                 dropout=0.1,
                 heads=1):
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        self.decoder_start_index = decoder_start_index
        self.decoder_end_index = decoder_end_index

        self.encoder = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder = nn.Embedding(decoder_vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers, dim_feedforward=d_fft,
                                          dropout=dropout, activation='relu')
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

    def forward(self, src, trg):         # shape: [N, T]

        if self.training:
            self.step += 1

        src = src.transpose(0, 1)        # shape: [T, N]
        trg = trg.transpose(0, 1)

        trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = self.make_len_mask(src).to(trg.device)
        trg_pad_mask = self.make_len_mask(trg).to(trg.device)

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        trg = self.decoder(trg) * math.sqrt(self.d_model)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=trg_mask,
                                  memory_mask=self.memory_mask, src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)
        output = output.transpose(0, 1)
        return output

    def generate(self,
                 input: torch.tensor,          # shape: [N, T]
                 max_len=100) -> torch.tensor:

        input = input.transpose(0, 1)          # shape: [T, N]
        src_pad_mask = self.make_len_mask(input).to(input.device)
        input = self.encoder(input) * math.sqrt(self.d_model)
        input = self.pos_encoder(input)
        input = self.transformer.encoder(input, src_key_padding_mask=src_pad_mask)

        out_indices = [self.decoder_start_index]
        for i in range(max_len):
            trg_tensor = torch.tensor(out_indices).long().unsqueeze(1).to(input.device)

            output = self.decoder(trg_tensor) * math.sqrt(self.d_model)
            output = self.pos_decoder(output)
            output = self.transformer.decoder(output, input, memory_key_padding_mask=src_pad_mask)
            output = self.fc_out(output)
            out_token = output.argmax(2)[-1].item()
            out_indices.append(out_token)
            if out_token == self.decoder_end_index:
                break

        return out_indices

    def get_step(self):
        return self.step.data.item()

    @classmethod
    def from_config(cls, config: dict) -> 'TransformerModel':
        return TransformerModel(
            encoder_vocab_size=config['encoder_vocab_size'],
            decoder_vocab_size=config['decoder_vocab_size'],
            decoder_start_index=config['decoder_start_index'],
            decoder_end_index=config['decoder_end_index'],
            d_model=config['d_model'],
            d_fft=config['d_fft'],
            encoder_layers=config['encoder_layers'],
            decoder_layers=config['decoder_layers'],
            dropout=config['dropout'],
            heads=config['heads']
        )

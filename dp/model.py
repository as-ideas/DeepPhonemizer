import math
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder, ModuleList
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dp.text import Preprocessor


class LstmModel(torch.nn.Module):

    def __init__(self,
                 num_symbols_in: int,
                 num_symbols_out: int,
                 lstm_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.register_buffer('step', torch.tensor(1, dtype=torch.int))
        self.embedding = nn.Embedding(num_embeddings=num_symbols_in, embedding_dim=lstm_dim)
        lstms = [torch.nn.LSTM(lstm_dim, lstm_dim, batch_first=True, bidirectional=True)]
        for i in range(1, num_layers):
            lstms.append(
                torch.nn.LSTM(2 * lstm_dim, lstm_dim, batch_first=True, bidirectional=True)
            )
        self.lstms = ModuleList(lstms)
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols_out)

    def forward(self,
                x: torch.tensor,
                x_len: torch.tensor = None) -> torch.tensor:
        if self.training:
            self.step += 1
        x = self.embedding(x)
        if x_len is not None:
            x = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        for lstm in self.lstms:
            x, _ = lstm(x)
        if x_len is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.lin(x)
        return x

    def get_step(self) -> int:
        return self.step.data.item()

    @classmethod
    def from_config(cls, config: dict) -> 'LstmModel':
        preprocessor = Preprocessor.from_config(config)
        model = LstmModel(
            num_symbols_in=preprocessor.text_tokenizer.vocab_size,
            num_symbols_out=preprocessor.phoneme_tokenizer.vocab_size,
            lstm_dim=config['model']['lstm_dim'],
            num_layers=config['model']['num_layers']
        )
        return model


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=5000) -> None:
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

    def forward(self, x: torch.tensor) -> torch.tensor:         # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=1024,
                 layers=4,
                 dropout=0.1,
                 heads=1) -> None:
        super(TransformerModel, self).__init__()

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

    def generate_square_subsequent_mask(self, sz: int) -> torch.tensor:
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp: torch.tensor) -> torch.tensor:
        return (inp == 0).transpose(0, 1)

    def forward(self, x, **kwargs) -> torch.tensor:         # shape: [N, T]

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

    def get_step(self) -> int:
        return self.step.data.item()

    @classmethod
    def from_config(cls, config: dict) -> 'TransformerModel':
        preprocessor = Preprocessor.from_config(config)
        return TransformerModel(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


def load_checkpoint(checkpoint_path: str, device='cpu') -> Tuple[torch.nn.Module, Dict[str, Any]]:
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint['config']['model']['type']
    supported_types = ['lstm', 'transformer']
    if model_type == 'lstm':
            model = LstmModel.from_config(checkpoint['config']).to(device)
    elif model_type == 'transformer':
            model = TransformerModel.from_config(checkpoint['config']).to(device)
    else:
        raise ValueError(f'Model type not supported: {model_type}. Supported types: {supported_types}')

    model.load_state_dict(checkpoint['model'])

    model.eval()
    return model, checkpoint

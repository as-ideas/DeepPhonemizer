from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder, ModuleList
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dp.model.utils import get_dedup_tokens, make_len_mask, generate_square_subsequent_mask, PositionalEncoding
from dp.preprocessing.text import Preprocessor


class ModelType(Enum):
    LSTM_MODEL = 'lstm_model'
    TRANSFORMER = 'transformer'
    AUTOREG_TRANSFORMER = 'autoreg_transformer'

    def is_autoregressive(self) -> bool:
        return self in {ModelType.AUTOREG_TRANSFORMER}


class Model(torch.nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, batch: Dict[str, torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
        """
        Generates phonemes for a text batch

        :param batch: Dictionary containing: 'text' (tokenized text tensor),
                     'text_len' (text length tensor for LstmModel),
                     'start_index' (phoneme start indices for AutoregressiveTransformer)
        :return: Tuple, where the first element is a tensor (phoneme tokens) and the second element
                 is a tensor (phoneme token probabilities)
        """
        pass


class LstmModel(Model):

    def __init__(self,
                 num_symbols_in: int,
                 num_symbols_out: int,
                 lstm_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_symbols_in, embedding_dim=lstm_dim)
        lstms = [torch.nn.LSTM(lstm_dim, lstm_dim, batch_first=True, bidirectional=True)]
        for i in range(1, num_layers):
            lstms.append(
                torch.nn.LSTM(2 * lstm_dim, lstm_dim, batch_first=True, bidirectional=True)
            )
        self.lstms = ModuleList(lstms)
        self.lin = torch.nn.Linear(2 * lstm_dim, num_symbols_out)

    def forward(self,
                batch: Dict[str, torch.tensor]) -> torch.tensor:
        x = batch['text']
        x_len = batch['text_len']
        x = self.embedding(x)
        if x_len is not None:
            x = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        for lstm in self.lstms:
            x, _ = lstm(x)
        if x_len is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.lin(x)
        return x

    def generate(self,
                 batch: Dict[str, torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
        with torch.no_grad():
            x = self.forward(batch)
        tokens, logits = get_dedup_tokens(x)
        return tokens, logits

    @classmethod
    def from_config(cls, config: dict) -> 'LstmModel':
        preprocessor = Preprocessor.from_config(config)
        model = LstmModel(
            num_symbols_in=preprocessor.text_tokenizer.vocab_size,
            num_symbols_out=preprocessor.phoneme_tokenizer.vocab_size,
            lstm_dim=config['model']['lstm_dim'],
            num_layers=config['model']['layers']
        )
        return model


class ForwardTransformer(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=1024,
                 layers=4,
                 dropout=0.1,
                 heads=1) -> None:
        super(ForwardTransformer, self).__init__()

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

    def forward(self,
                batch: Dict[str, torch.tensor]) -> torch.tensor:         # shape: [N, T]

        x = batch['text']
        x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x

    def generate(self,
                 batch: Dict[str, torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
        with torch.no_grad():
            x = self.forward(batch)
        tokens, logits = get_dedup_tokens(x)
        return tokens, logits

    @classmethod
    def from_config(cls, config: dict) -> 'ForwardTransformer':
        preprocessor = Preprocessor.from_config(config)
        return ForwardTransformer(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


class AutoregressiveTransformer(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 end_index: int,
                 d_model=512,
                 d_fft=1024,
                 encoder_layers=4,
                 decoder_layers=4,
                 dropout=0.1,
                 heads=1):
        super(AutoregressiveTransformer, self).__init__()

        self.end_index = end_index

        self.d_model = d_model

        self.encoder = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.decoder = nn.Embedding(decoder_vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers, dim_feedforward=d_fft,
                                          dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self, batch: Dict[str, torch.tensor]):         # shape: [N, T]
        src = batch['text']
        trg = batch['phonemes'][:, :-1]

        src = src.transpose(0, 1)        # shape: [T, N]
        trg = trg.transpose(0, 1)

        trg_mask = generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = make_len_mask(src).to(trg.device)
        trg_pad_mask = make_len_mask(trg).to(trg.device)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=None, tgt_mask=trg_mask,
                                  memory_mask=None, src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)
        output = output.transpose(0, 1)
        return output

    def generate(self,
                 batch: Dict[str, torch.tensor],
                 max_len=100) -> Tuple[torch.tensor, torch.tensor]:

        input = batch['text']
        start_index = batch['start_index']

        batch_size = input.size(0)
        input = input.transpose(0, 1)          # shape: [T, N]
        src_pad_mask = make_len_mask(input).to(input.device)
        with torch.no_grad():
            input = self.encoder(input)
            input = self.pos_encoder(input)
            input = self.transformer.encoder(input,
                                             src_key_padding_mask=src_pad_mask)
            out_indices = start_index.unsqueeze(0)
            out_logits = []
            for i in range(max_len):
                tgt_mask = generate_square_subsequent_mask(i + 1).to(input.device)
                output = self.decoder(out_indices)
                output = self.pos_decoder(output)
                output = self.transformer.decoder(output,
                                                  input,
                                                  memory_key_padding_mask=src_pad_mask,
                                                  tgt_mask=tgt_mask)
                output = self.fc_out(output)  # shape: [T, N, V]
                out_tokens = output.argmax(2)[-1:, :]
                out_logits.append(output[-1:, :, :])

                out_indices = torch.cat([out_indices, out_tokens], dim=0)
                stop_rows, _ = torch.max(out_indices == self.end_index, dim=0)
                if torch.sum(stop_rows) == batch_size:
                    break

        out_indices = out_indices.transpose(0, 1)  # out shape [N, T]
        out_logits = torch.cat(out_logits, dim=0).transpose(0, 1) # out shape [N, T, V]
        out_logits = out_logits.softmax(-1)
        out_probs = torch.ones((out_indices.size(0), out_indices.size(1)))
        for i in range(out_indices.size(0)):
            for j in range(0, out_indices.size(1)-1):
                out_probs[i, j+1] = out_logits[i, j].max()
        return out_indices, out_probs

    @classmethod
    def from_config(cls, config: dict) -> 'AutoregressiveTransformer':
        preprocessor = Preprocessor.from_config(config)
        return AutoregressiveTransformer(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            end_index=preprocessor.phoneme_tokenizer.end_index,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            encoder_layers=config['model']['layers'],
            decoder_layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


def create_model(model_type: ModelType, config: Dict[str, Any]) -> Model:
    if model_type is ModelType.LSTM_MODEL:
        model = LstmModel.from_config(config)
    elif model_type is ModelType.TRANSFORMER:
        model = ForwardTransformer.from_config(config)
    elif model_type is ModelType.AUTOREG_TRANSFORMER:
        model = AutoregressiveTransformer.from_config(config)
    else:
        raise ValueError(f'Unsupported model type: {model_type}. '
                         f'Supported types: {[t.value for t in ModelType]}')
    return model


def load_checkpoint(checkpoint_path: str, device='cpu') -> Tuple[Model, Dict[str, Any]]:
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint['config']['model']['type']
    model_type = ModelType(model_type)
    model = create_model(model_type, config=checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint

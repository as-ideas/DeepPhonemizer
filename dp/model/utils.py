import math
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

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


def get_dedup_tokens(logits_batch: torch.tensor) \
        -> Tuple[torch.tensor, torch.tensor]:

    """ Returns deduplicated tokens and probs of tokens """

    logits_batch = logits_batch.softmax(-1)
    out_tokens, out_probs = [], []
    for i in range(logits_batch.size(0)):
        logits = logits_batch[i]
        max_logits, max_indices = torch.max(logits, dim=-1)
        max_logits = max_logits[max_indices!=0]
        max_indices = max_indices[max_indices!=0]
        cons_tokens, counts = torch.unique_consecutive(
            max_indices, return_counts=True)
        out_probs_i = []
        ind = 0
        for c in counts:
            max_logit = max_logits[ind:ind + c].max()
            out_probs_i.append(max_logit.item())
            ind = ind + c
        out_tokens.append(cons_tokens)
        out_probs_i = torch.tensor(out_probs_i)
        out_probs.append(out_probs_i)

    out_tokens = pad_sequence(out_tokens, batch_first=True, padding_value=0)
    out_probs = pad_sequence(out_probs, batch_first=True, padding_value=0)

    return out_tokens, out_probs


def generate_square_subsequent_mask(sz: int) -> torch.tensor:
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def make_len_mask(inp: torch.tensor) -> torch.tensor:
    return (inp == 0).transpose(0, 1)


def get_len_util_stop(sequence: torch.tensor, end_index: int) -> torch.tensor:
    for i, val in enumerate(sequence):
        if val == end_index:
            return i + 1
    return len(sequence)


def trim_util_stop(sequence: torch.tensor, end_index: int) -> torch.tensor:
    seq_len = get_len_util_stop(sequence, end_index)
    return sequence[:seq_len]

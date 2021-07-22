import math
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=5000) -> None:
        """
        Initializes positional encoding.

        Args:
            d_model (int): Dimension of model.
            dropout (float): Dropout after positional encoding.
            max_len: Max length of precalculated position sequence.
        """

        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


def get_dedup_tokens(logits_batch: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts a batch of logits into the batch most probable tokens and their probabilities.

    Args:
      logits_batch (Tensor): Batch of logits (N x T x V).

    Returns:
      Tuple: Deduplicated tokens. The first element is a tensor (token indices) and the second element
      is a tensor (token probabilities)

    """

    logits_batch = logits_batch.softmax(-1)
    out_tokens, out_probs = [], []
    for i in range(logits_batch.size(0)):
        logits = logits_batch[i]
        max_logits, max_indices = torch.max(logits, dim=-1)
        max_logits = max_logits[max_indices!=0]
        max_indices = max_indices[max_indices!=0]
        cons_tokens, counts = torch.unique_consecutive(
            max_indices, return_counts=True)
        out_probs_i = torch.zeros(len(counts), device=logits.device)
        ind = 0
        for i, c in enumerate(counts):
            max_logit = max_logits[ind:ind + c].max()
            out_probs_i[i] = max_logit
            ind = ind + c
        out_tokens.append(cons_tokens)
        out_probs.append(out_probs_i)

    out_tokens = pad_sequence(out_tokens, batch_first=True, padding_value=0.).long()
    out_probs = pad_sequence(out_probs, batch_first=True, padding_value=0.)

    return out_tokens, out_probs


def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def _make_len_mask(inp: torch.Tensor) -> torch.Tensor:
    return (inp == 0).transpose(0, 1)


def _get_len_util_stop(sequence: torch.Tensor, end_index: int) -> int:
    for i, val in enumerate(sequence):
        if val == end_index:
            return i + 1
    return len(sequence)


def _trim_util_stop(sequence: torch.Tensor, end_index: int) -> torch.Tensor:
    seq_len = _get_len_util_stop(sequence, end_index)
    return sequence[:seq_len]

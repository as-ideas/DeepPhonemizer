import pickle
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Iterable, Sized

import torch
import yaml
import math

from torch.nn.utils.rnn import pad_sequence


def read_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, 'w+', encoding='utf-8') as stream:
        yaml.dump(config, stream, default_flow_style=False)


def get_files(path: str, extension='.wav') -> List[Path]:
    return list(Path(path).expanduser().resolve().rglob(f'*{extension}'))


def pickle_binary(data: object, file: Union[str, Path]) -> None:
    with open(str(file), 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]) -> Any:
    with open(str(file), 'rb') as f:
        return pickle.load(f)


def to_device(batch: Dict[str, torch.tensor], device: torch.device) -> Dict[str, torch.tensor]:
    return {key: val.to(device) for key, val in batch.items()}


def get_sequence_prob(probs: torch.tensor) -> float:
    if probs is None or len(probs) == 0:
        return 0.
    if 0 in probs:
        return 0
    prob = math.exp(sum([math.log(p) for p in probs]))
    return prob


def batchify(input: list, batch_size: int) -> List[list]:
    l = len(input)
    output = []
    for i in range(0, l, batch_size):
        batch = input[i:min(i + batch_size, l)]
        output.append(batch)
    return output


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

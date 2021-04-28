import math
import pickle
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple

import torch
import yaml


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


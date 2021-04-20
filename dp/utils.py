import pickle
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Iterable, Sized

import torch
import yaml
import math

from dp.model import TransformerModel
from dp.text import Preprocessor


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


def get_sequence_prob(tokens: List[int], logits: torch.tensor) -> float:
    if len(tokens) == 0:
        return 1.
    norm_logits = logits.softmax(dim=-1)
    probs = [norm_logits[i, p] for i, p in enumerate(tokens[1:])]
    prob = math.exp(sum([math.log(p) for p in probs]))
    return prob


def load_checkpoint(checkpoint_path: str, device='cpu') -> Tuple[TransformerModel, Dict[str, Any]]:
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerModel.from_config(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def batchify(input: list, batch_size: int) -> List[list]:
    l = len(input)
    output = []
    for i in range(0, l, batch_size):
        batch = input[i:min(i + batch_size, l)]
        output.append(batch)
    return output


import pickle
from pathlib import Path
from typing import Dict, List, Any, Union

import torch
import yaml


def read_config(path: str) -> Dict[str, Any]:
    """
    Reads the config dictionary from the yaml file.

    Args:
        path (str): Path to the .yaml file.

    Returns:
        Dict[str, Any]: Configuration.

    """

    with open(path, 'r', encoding='utf-8') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Saves the config as a yaml file.

    Args:
        config (Dict[str, Any]): Configuration.
        path (str): Path to save the dictionary to (.yaml).
    """

    with open(path, 'w+', encoding='utf-8') as stream:
        yaml.dump(config, stream, default_flow_style=False)


def get_files(path: str, extension: str = '.wav') -> List[Path]:
    """
    Recursively retrieves all files with a given extension from a folder.

    Args:
      path (str): Path to the folder to retrieve files from.
      extension (str): Extension of files to be retrieved (Default value = '.wav').

    Returns:
        List[Path]: List of paths to the found files.
    """

    return list(Path(path).expanduser().resolve().rglob(f'*{extension}'))


def pickle_binary(data: object, file: Union[str, Path]) -> None:
    """
    Pickles a given object to a binary file.

    Args:
        data (object): Object to be pickled.
        file (Union[str, Path]): Path to destination file (use the .pkl extension).
    """

    with open(str(file), 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]) -> object:
    """
    Unpickles a given binary file to an object

    Args:
        file (nion[str, Path]): Path to the file.

    Returns:
        object: Unpickled object.

    """

    with open(str(file), 'rb') as f:
        return pickle.load(f)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Sends a batch of data to the given torch devicee (cpu or cuda).

    Args:
        batch (Dict[str, torch.Tensor]): Batch to be send to the device.
        device (torch.device): Device (either torch.device('cpu') or torch.device('cuda').

    Returns:
        Dict[str, torch.Tensor]: The batch at the given device.

    """

    return {key: val.to(device) for key, val in batch.items()}
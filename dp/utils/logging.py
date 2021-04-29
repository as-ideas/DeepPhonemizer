import logging
import logging.config

import math
import pickle
from logging import INFO, Logger, getLogger
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple

import torch
import yaml

from dp.utils.io import read_config

CONFIG_FILE = 'dp/configs/logging.yaml'
config = read_config(CONFIG_FILE)
logging.config.dictConfig(config)


def get_logger(name: str) -> Logger:
    logger = getLogger(name)
    return logger
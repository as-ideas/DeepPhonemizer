import logging.config
import logging.config
import os
from logging import Logger, getLogger
from pathlib import Path

import dp.configs
from dp.utils.io import read_config

main_dir = os.path.dirname(os.path.abspath(dp.configs.__file__))
config_file_path = Path(main_dir) / 'logging.yaml'
config = read_config(config_file_path)
logging.config.dictConfig(config)


def get_logger(name: str) -> Logger:
    logger = getLogger(name)
    return logger
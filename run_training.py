import logging.config
from pathlib import Path
from dp.preprocess import preprocess
from dp.train import train
from dp.utils.io import read_config

config_file_path = Path('logging.yaml')
config = read_config(config_file_path)
logging.config.dictConfig(config)

if __name__ == '__main__':

    train_data = [('de', 'gewürz', '123123')] * 1000

    val_data = [('de', 'benützten', '121212123')] * 100

    config_file = 'dp/configs/autoreg_config.yaml'

    preprocess(config_file=config_file,
               train_data=train_data,
               val_data=val_data,
               deduplicate_train_data=False)

    num_gpus = 0

    train(rank=0, num_gpus=num_gpus, config_file=config_file)
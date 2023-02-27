import logging.config
from pathlib import Path
import torch
import torch.multiprocessing as mp

from dp.preprocess import preprocess
from dp.train import train
from dp.utils.io import read_config

config_file_path = Path('logging.yaml')
config = read_config(config_file_path)
logging.config.dictConfig(config)

if __name__ == '__main__':

    train_data = [('en_us', 'young', 'jʌŋ'),
                  ('de', 'benützten', 'bənʏt͡stn̩'),
                  ('de', 'gewürz', 'ɡəvʏʁt͡s')] * 1000

    val_data = [('en_us', 'young', 'jʌŋ'),
                ('de', 'benützten', 'bənʏt͡stn̩')] * 100

    config_file = 'dp/configs/forward_config.yaml'

    preprocess(config_file=config_file,
               train_data=train_data,
               val_data=val_data,
               deduplicate_train_data=False)

    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(num_gpus, config_file))
    else:
        train(rank=0, num_gpus=num_gpus, config_file=config_file)
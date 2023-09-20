import logging.config
from pathlib import Path
from random import Random

import numpy as np
from tqdm import tqdm

from dp.preprocess import preprocess
from dp.train import train
from dp.utils.io import read_config, unpickle_binary

config_file_path = Path('logging.yaml')
config = read_config(config_file_path)
logging.config.dictConfig(config)

if __name__ == '__main__':
    config_file = 'dp/configs/autoreg_config.yaml'
    config = read_config(config_file)

    speaker_dict: dict = unpickle_binary('/Users/cschaefe/datasets/multispeaker_all_data/speaker_dict.pkl')
    text_dict = unpickle_binary('/Users/cschaefe/datasets/multispeaker_all_data/text_dict.pkl')
    speakers = sorted(list(set(speaker_dict.values())))
    dur_path = Path('/Users/cschaefe/datasets/multispeaker_all_data/alg')
    config['preprocessing']['languages'] = speakers

    train_data = []

    for index, (id, phons) in tqdm(enumerate(text_dict.items()), total=len(text_dict)):
        dur = np.load(dur_path / f'{id}.npy')
        phons = ''.join([p for p in phons if p in config['preprocessing']['text_symbols']])

        dur = [str(min(d, 9)) for d in dur]
        dur = ''.join(dur)

        if 3 < len(phons) < 300 and len(phons) == len(dur):
            speaker = speaker_dict[id]
            train_data.append((speaker, phons, dur))
            #print(train_data[-1])

        if index > 1000:
            break

    Random(42).shuffle(train_data)
    val_data = train_data[:124]
    train_data = train_data[124:]
    preprocess(config=config,
               train_data=train_data,
               val_data=val_data,
               deduplicate_train_data=False)

    num_gpus = 0

    train(rank=0, num_gpus=num_gpus, config=config)
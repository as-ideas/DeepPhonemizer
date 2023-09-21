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

def norm_pitch(pitch):
    p = pitch
    pitch = min(pitch, 3)
    pitch = max(pitch, -3)
    pitch = (pitch / 3) * 128
    pitch = int(pitch)
    return pitch

if __name__ == '__main__':
    config_file = 'dp/configs/autoreg_config.yaml'
    config = read_config(config_file)

    speaker_dict: dict = unpickle_binary('/Users/cschaefe/datasets/multispeaker_all_data/speaker_dict.pkl')
    text_dict = unpickle_binary('/Users/cschaefe/datasets/multispeaker_all_data/text_dict.pkl')
    speakers = sorted(list(set(speaker_dict.values())))
    pitch_path = Path('/Users/cschaefe/datasets/multispeaker_all_data/phon_pitch')
    config['preprocessing']['languages'] = speakers
    config['preprocessing']['phoneme_symbols'] = [str(i) for i in range(-128, 129)]

    train_data = []

    for index, (id, phons) in tqdm(enumerate(text_dict.items()), total=len(text_dict)):
        pitch = np.load(pitch_path / f'{id}.npy')
        phons = ''.join([p for p in phons if p in config['preprocessing']['text_symbols']])

        pitch = [str(norm_pitch(p)) for p in pitch]
        #pitch = ''.join(pitch)

        if 3 < len(phons) < 300 and len(phons) == len(pitch):
            speaker = speaker_dict[id]
            train_data.append((speaker, phons, pitch))
            print(train_data[-1])

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
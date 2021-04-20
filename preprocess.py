from multiprocessing.pool import Pool
from pathlib import Path
from random import Random
from typing import Iterable, Tuple

import tqdm
import pickle
import argparse

from dp.text import Preprocessor
from dp.utils import read_config, pickle_binary, unpickle_binary


def get_data(file: str):
    with open(file, 'rb') as f:
        df = pickle.load(f)
    tuples = df[['title', 'pronunciation']]
    tuples = [tuple(x) for x in tuples.to_numpy()]
    data_set = {w for w, _ in tuples}
    train_data = []
    max_len = 50
    all_phons = set()
    for word, phon in tuples:
        all_phons.update(set(phon))
        if 0 < len(phon) < max_len and ' ' not in word and 0 < len(word) < max_len:
            train_data.append(('de', word, phon))
            if word.lower() not in data_set:
                word_ = word.lower()
                #train_data.append(('de', word_, phon))
            if word.title() not in data_set:
                word_ = word.title()
                #train_data.append(('de', word_, phon))

    return train_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--checkpoint', '-cp', default=None, help='Points to the a model file to restore.')
    parser.add_argument('--path', '-p', help='Points to the a file with data.')

    args = parser.parse_args()

    config = read_config(args.config)
    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)

    #raw_data = get_data(args.path)
    raw_data = unpickle_binary(args.path)
    languages = set(config['preprocessing']['languages'])
    raw_data = [r for r in raw_data if r[0] in languages]

    raw_data.sort()

    random = Random(42)
    random.shuffle(raw_data)

    train_data = raw_data[config['preprocessing']['n_val']:]
    val_data = raw_data[:config['preprocessing']['n_val']]

    preprocessor = Preprocessor.from_config(config)

    train_dataset = []
    for i, (lang, text, phonemes) in enumerate(tqdm.tqdm(train_data, total=len(train_data))):
        tokens = preprocessor((lang, text, phonemes))
        train_dataset.append(tokens)

    val_dataset = []
    for i, (lang, text, phonemes) in enumerate(tqdm.tqdm(val_data, total=len(val_data))):
        tokens = preprocessor((lang, text, phonemes))
        val_dataset.append(tokens)

    print('saving datasets...')
    pickle_binary(train_dataset, data_dir / 'train_dataset.pkl')
    pickle_binary(val_dataset, data_dir / 'val_dataset.pkl')

    phoneme_dictionary = dict()
    for lang, text, phoneme in raw_data:
        lang_dict = phoneme_dictionary.get(lang, {})
        lang_dict[text] = phoneme
        phoneme_dictionary[lang] = lang_dict

    pickle_binary(phoneme_dictionary, data_dir / 'phoneme_dict.pkl')

    with open(data_dir / 'phoneme_list.txt', 'w+', encoding='utf-8') as f:
        for lang, text, phoneme in raw_data:
            f.write(f'{lang}\t{text}\t{phoneme}\n')


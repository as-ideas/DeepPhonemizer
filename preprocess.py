from collections import Counter
from pathlib import Path
import argparse
from pathlib import Path
from random import Random

import tqdm

from dp.text import Preprocessor
from dp.utils import read_config, pickle_binary, unpickle_binary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepPhonemizer')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    args = parser.parse_args()
    config = read_config(args.config)
    languages = set(config['preprocessing']['languages'])

    train_file = config['paths']['train_file']
    val_file = config['paths']['val_file']
    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reading train data from {train_file}')
    train_data = unpickle_binary(train_file)
    train_data = [r for r in train_data if r[0] in languages]

    if val_file is not None:
        print(f'Reading val data from {val_file}')
        val_data = unpickle_binary(val_file)
        val_data = [r for r in val_data if r[0] in languages]
    else:
        n_val = config['preprocessing']['n_val']
        print(f'Performing random split with num val: {n_val}')
        train_data.sort()
        random = Random(42)
        random.shuffle(train_data)
        train_data = train_data[n_val:]
        val_data = train_data[:n_val]

    preprocessor = Preprocessor.from_config(config)

    train_count = Counter()
    val_count = Counter()

    print('Processing data...')
    train_dataset = []
    for i, (lang, text, phonemes) in enumerate(tqdm.tqdm(train_data, total=len(train_data))):
        tokens = preprocessor((lang, text, phonemes))
        train_dataset.append(tokens)
        train_count.update([lang])

    val_dataset = []
    for i, (lang, text, phonemes) in enumerate(val_data):
        tokens = preprocessor((lang, text, phonemes))
        val_dataset.append(tokens)
        val_count.update([lang])

    print(f'\nSaving datasets to: {data_dir.absolute()}')
    pickle_binary(train_dataset, data_dir / 'train_dataset.pkl')
    pickle_binary(val_dataset, data_dir / 'val_dataset.pkl')
    phoneme_dictionary = dict()
    all_data = sorted(train_data + val_data)
    for lang, text, phoneme in all_data:
        lang_dict = phoneme_dictionary.get(lang, {})
        lang_dict[text] = phoneme
        phoneme_dictionary[lang] = lang_dict

    pickle_binary(phoneme_dictionary, data_dir / 'phoneme_dict.pkl')
    with open(data_dir / 'combined_dataset.txt', 'w+', encoding='utf-8') as f:
        for lang, text, phoneme in all_data:
            f.write(f'{lang}\t{text}\t{phoneme}\n')

    print(f'Preprocessing done. \nTrain counts: {train_count.most_common()}'
          f'\nVal counts: {val_count.most_common()}')
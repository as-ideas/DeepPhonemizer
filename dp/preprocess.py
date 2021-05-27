from collections import Counter
from pathlib import Path
from random import Random
from typing import List, Tuple, Iterable

import tqdm

from dp.model.model import ModelType
from dp.preprocessing.text import Preprocessor
from dp.utils.io import read_config, pickle_binary
from dp.utils.logging import get_logger

logger = get_logger(__name__)


def preprocess(config_file: str,
               train_data: List[Tuple[str, Iterable[str], Iterable[str]]],
               val_data: List[Tuple[str, Iterable[str], Iterable[str]]] = None,
               deduplicate_train_data=True) -> None:

    config = read_config(config_file)

    model_type = config['model']['type']
    model_type = ModelType(model_type)
    if model_type.is_autoregressive() and config['preprocessing']['char_repeats'] > 1:
        char_repeats = config['preprocessing']['char_repeats']
        logger.warning(f'WARNING: You are training autoregressive model with char_repeats={char_repeats}. '
                       f'It is recommended to set char_repeats=1 in the config and preprocess again.')

    languages = set(config['preprocessing']['languages'])

    logger.info(f'Preprocessing, train data: with {len(train_data)} files.')

    data_dir = Path(config['paths']['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)

    train_dict = {(l, w): [] for l, w, p in train_data}
    for l, w, p in train_data:
        train_dict[(l, w)] = train_dict[(l, w)] + [(l, w, p)]
    train_keys = sorted(list(train_dict.keys()))

    if val_data is not None:
        val_data = [(l, w, p) for l, w, p in val_data if l in languages]
    else:
        n_val = config['preprocessing']['n_val']
        logger.info(f'Performing random split with num val: {n_val}')
        random = Random(42)
        random.shuffle(train_keys)
        val_keys = train_keys[:n_val]
        train_keys = train_keys[n_val:]
        val_data = []
        for k in val_keys:
            val_data.extend(train_dict[k])

    train_data = []
    for key in train_keys:
        data_list = train_dict[key]
        if deduplicate_train_data:
            train_data.append(data_list[0])
        else:
            train_data.extend(data_list)

    preprocessor = Preprocessor.from_config(config)

    train_count = Counter()
    val_count = Counter()

    logger.info('Processing train data...')
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

    logger.info(f'\nSaving datasets to: {data_dir.absolute()}')
    pickle_binary(train_dataset, data_dir / 'train_dataset.pkl')
    pickle_binary(val_dataset, data_dir / 'val_dataset.pkl')
    phoneme_dictionary = dict()

    all_data = []
    text_symbols = set(config['preprocessing']['text_symbols'])
    phoneme_symbols = set(config['preprocessing']['phoneme_symbols'])
    for lang, text, phon in sorted(train_data + val_data):
        text = ''.join([t for t in text if t in text_symbols])
        phons = ''.join([p for p in phon if p in phoneme_symbols])
        all_data.append((lang, text, phons))

    for l, w, p in all_data:
        lang_dict = phoneme_dictionary.setdefault(l, {})
        if w not in lang_dict:
            lang_dict[w] = p

    pickle_binary(phoneme_dictionary, data_dir / 'phoneme_dict.pkl')
    with open(data_dir / 'combined_dataset.txt', 'w+', encoding='utf-8') as f:
        for lang, text, phoneme in all_data:
            f.write(f'{lang}\t{text}\t{phoneme}\n')

    logger.info(f'Preprocessing. \nTrain counts (deduplicated): {train_count.most_common()}'
                f'\nVal counts (including duplicates): {val_count.most_common()}')

    assert len(train_count) > 0, 'Preprocessing resulted in zero train counts!'
    assert len(val_count) > 0, 'Preprocessing resulted in zero validation counts!'

import pickle
import argparse
import random
from collections import Counter
from random import Random
from typing import List, Tuple, Dict, Iterable, Any

import tqdm
import torch
from dp.dataset import new_dataloader
from dp.model import TransformerModel
from dp.text import Tokenizer, Preprocessor
from dp.trainer import Trainer
from dp.utils import read_config, pickle_binary


# replace that later, copied from headliner, yes its BS
def get_data(file: str):
    with open(file, 'rb') as f:
        df = pickle.load(f)
    tuples = df[['title', 'pronunciation']]
    tuples = [tuple(x) for x in tuples.to_numpy()]
    data_set = {w for w, _ in tuples}
    train_data = []
    max_len = 50
    all_phons = set()
    for word, phon in tqdm.tqdm(tuples, total=len(tuples)):
        all_phons.update(set(phon))
        if 0 < len(phon) < max_len and ' ' not in word and 0 < len(word) < max_len:
            train_data.append(('de', word, phon))
            if word.lower() not in data_set:
                word_ = word.lower()
                train_data.append(('de', word_, phon))
            if word.title() not in data_set:
                word_ = word.title()
                train_data.append(('de', word_, phon))

    return train_data


def init_checkpoint(raw_data: List[Tuple[str, Iterable[str], Iterable[str]]], config: Dict[str, Any]) -> Dict[str, Any]:
    preprocessor = Preprocessor.build_from_data(data=raw_data, lowercase=config['preprocessing']['lowercase'])
    config['model']['encoder_vocab_size'] = preprocessor.text_tokenizer.vocab_size
    config['model']['decoder_vocab_size'] = preprocessor.phoneme_tokenizer.vocab_size
    config['model']['decoder_start_index'] = preprocessor.phoneme_tokenizer.start_index
    config['model']['decoder_end_index'] = preprocessor.phoneme_tokenizer.end_index
    checkpoint = {
        'preprocessor': preprocessor,
        'config': config,
    }
    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--checkpoint', '-cp', default=None, help='Points to the a model file to restore.')
    args = parser.parse_args()

    raw_data = get_data('/home/sysgen/chris/data/heavily_cleaned_phoneme_dataset_DE.pkl')
    #raw_data = get_data('/Users/cschaefe/datasets/nlp/heavily_cleaned_phoneme_dataset_DE.pkl')

    config = read_config(args.config)

    if args.checkpoint:
        print(f'Restoring model from checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model = TransformerModel.from_config(checkpoint['config']['model'])
        model.load_state_dict(checkpoint['model'])
        print(f'Loaded model with step: {model.get_step()}')
        for key, val in config['training'].items():
            val_orig = checkpoint['config']['training'][key]
            if val_orig != val:
                print(f'Overwriting training param: {key} {val_orig} --> {val}')
                checkpoint['config']['training'][key] = val
        config = checkpoint['config']
    else:
        print('Initializing new model from config, build preprocessor...')
        checkpoint = init_checkpoint(raw_data=raw_data, config=config)
        model = TransformerModel.from_config(config['model'])

    print('Preprocessing...')
    preprocessor = checkpoint['preprocessor']
    random = random.Random(42)
    random.shuffle(raw_data)
    n_val = config['preprocessing']['n_val']
    train_data, val_data = raw_data[n_val:], raw_data[:n_val]

    # data augmentation, redo later
    train_data_concat = []
    for (l, w1, p1), (_, w2, p2) in zip(train_data[:-1], train_data[1:]):
        train_data_concat.append((l, w1, p1))
        train_data_concat.append((l, w1 + w2, p1 + p2))

    train_data_concat = preprocessor(train_data_concat)
    val_data = preprocessor(val_data)
    print('Training...')
    trainer = Trainer(checkpoint_dir=config['paths']['checkpoint_dir'])
    trainer.train(model=model, checkpoint=checkpoint,
                  train_data=train_data_concat, val_data=val_data)
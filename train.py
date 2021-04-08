import pickle
import argparse
import random
from collections import Counter
from typing import List, Tuple

import tqdm
import torch
from dp.dataset import new_dataloader
from dp.model import TransformerModel
from dp.text import Tokenizer
from dp.trainer import Trainer
from dp.utils import read_config


# replace that later
def get_data(file: str) -> List[Tuple[str, str, str]]:
    data = []
    with open(file, 'rb') as f:
        df = pickle.load(f)
    tuples = df[['title', 'pronunciation']]
    for x in tuples.to_numpy():
        data.append(('de', x[0], x[1]))
    return data


def get_symbols(raw_data: List[tuple]) -> Tuple[list, list, list]:
    lang_counter, text_counter, phoneme_counter = Counter(), Counter(), Counter()
    for lang, text, phonemes in raw_data:
        lang_counter.update([lang])
        text_counter.update(text)
        phoneme_counter.update(phonemes)
    text_symbols = sorted((list(text_counter.keys())))
    phoneme_symbols = sorted(list(phoneme_counter.keys()))
    lang_symbols = sorted(list(lang_counter.keys()))
    return lang_symbols, text_symbols, phoneme_symbols


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--checkpoint', '-cp', default=None, help='Points to the a model file to restore.')
    args = parser.parse_args()

    raw_data = get_data('/home/sysgen/chris/data/heavily_cleaned_phoneme_dataset_DE.pkl')

    config = read_config(args.config)

    if args.checkpoint:
        print(f'Restoring model from checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model = TransformerModel.from_config(checkpoint['config']['model'])
        print(f'Restored model with step {model.get_step()}')
        for key, val in config['training'].items():
            val_orig = checkpoint['config']['training'][key]
            if val_orig != val:
                print(f'Overwriting training config: {key} {val_orig} --> {val}')
                checkpoint['config']['training'][key] = val
        config = checkpoint['config']
    else:
        print('Initializing new model from config, build tokenizers...')
        lang_symbols, text_symbols, phoneme_symbols = get_symbols(raw_data)
        lang_indices = {l: i for i, l in enumerate(lang_symbols)}
        text_tokenizer = Tokenizer(text_symbols, lowercase=config['preprocessing']['lowercase'])
        phoneme_tokenizer = Tokenizer(phoneme_symbols, lowercase=config['preprocessing']['lowercase'])

        print('Creating model...')
        config['model']['encoder_vocab_size'] = text_tokenizer.vocab_size
        config['model']['decoder_vocab_size'] = phoneme_tokenizer.vocab_size
        config['model']['decoder_start_index'] = phoneme_tokenizer.start_index
        config['model']['decoder_end_index'] = phoneme_tokenizer.end_index
        model = TransformerModel.from_config(config['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lang_indices': lang_indices,
            'text_tokenizer': text_tokenizer,
            'phoneme_tokenizer': phoneme_tokenizer,
            'config': config,
            'data': raw_data
        }

    print('Tokenizing...')
    data = []
    for lang, text, phonemes in tqdm.tqdm(raw_data, total=len(raw_data)):
        text = text.lower()
        text_tokens = checkpoint['text_tokenizer'](text)
        phoneme_tokens = checkpoint['phoneme_tokenizer'](phonemes)
        lang_index = checkpoint['lang_indices'][lang]
        data.append((lang_index, text_tokens, phoneme_tokens))

    print('Train...')
    trainer = Trainer(checkpoint_dir=config['paths']['checkpoint_dir'])

    random = random.Random(42)
    random.shuffle(data)
    n_val = config['preprocessing']['n_val']
    train_data, val_data = data[n_val:], data[:n_val]
    trainer.train(model=model, checkpoint=checkpoint,
                  train_data=train_data, val_data=val_data)
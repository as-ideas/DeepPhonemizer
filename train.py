import pickle
import argparse
import random
from collections import Counter
from typing import List, Tuple, Dict

import tqdm
import torch
from dp.dataset import new_dataloader
from dp.model import TransformerModel
from dp.text import Tokenizer, Preprocessor
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
    data_filtered = []
    for lang, text, phon in data:
        if 0 < len(phon) < 50 and ' ' not in text and 0 < len(text) <50:
            data_filtered.append((lang, text, phon))

    return data_filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--checkpoint', '-cp', default=None, help='Points to the a model file to restore.')
    args = parser.parse_args()

    raw_data = get_data('/Users/cschaefe/datasets/nlp/heavily_cleaned_phoneme_dataset_DE.pkl')

    config = read_config(args.config)

    if args.checkpoint:
        print(f'Restoring model from checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model = TransformerModel.from_config(checkpoint['config']['model'])
        model.load_state_dict(checkpoint['model'])
        print(f'Restored model with step {model.get_step()}')
        for key, val in config['training'].items():
            val_orig = checkpoint['config']['training'][key]
            if val_orig != val:
                print(f'Overwriting training param: {key} {val_orig} --> {val}')
                checkpoint['config']['training'][key] = val
        config = checkpoint['config']
        preprocessor = checkpoint['preprocessor']
    else:
        print('Initializing new model from config, build preprocessor...')
        preprocessor = Preprocessor.build_from_data(data=raw_data)
        print('Creating model...')
        config['model']['encoder_vocab_size'] = preprocessor.text_tokenizer.vocab_size
        config['model']['decoder_vocab_size'] = preprocessor.phoneme_tokenizer.vocab_size
        config['model']['decoder_start_index'] = preprocessor.phoneme_tokenizer.start_index
        config['model']['decoder_end_index'] = preprocessor.phoneme_tokenizer.end_index
        model = TransformerModel.from_config(config['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'preprocessor': preprocessor,
            'config': config,
        }

    print('Preprocessing...')
    data = preprocessor(raw_data)
    random = random.Random(42)
    random.shuffle(data)
    n_val = config['preprocessing']['n_val']
    train_data, val_data = data[n_val:], data[:n_val]

    print('Training...')
    trainer = Trainer(checkpoint_dir=config['paths']['checkpoint_dir'])
    trainer.train(model=model, checkpoint=checkpoint,
                  train_data=train_data, val_data=val_data)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepForcedAligner.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--checkpoint', '-cp', default=None, help='Points to the a model file to restore.')
    args = parser.parse_args()

    config = read_config(args.config)

    if args.checkpoint:
        print(f'Restoring model from checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model = TransformerModel.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        print(f'Loaded model with step: {model.get_step()}')
        for key, val in config['training'].items():
            val_orig = checkpoint['config']['training'][key]
            if val_orig != val:
                print(f'Overwriting training param: {key} {val_orig} --> {val}')
                checkpoint['config']['training'][key] = val
        config = checkpoint['config']
    else:
        print('Initializing new model from config...')
        preprocessor = Preprocessor.from_config(config)
        model = TransformerModel.from_config(config)
        checkpoint = {
            'preprocessor': preprocessor,
            'config': config,
        }

    trainer = Trainer(checkpoint_dir=config['paths']['checkpoint_dir'])
    trainer.train(model=model, checkpoint=checkpoint)
import pickle
import argparse
import random
from collections import Counter
from random import Random
from typing import List, Tuple, Dict, Iterable, Any

import tqdm
import torch
from dp.dataset import new_dataloader
from dp.model import LstmModel, ForwardTransformer, AutoregressiveTransformer, load_checkpoint
from dp.text import SequenceTokenizer, Preprocessor
from dp.trainer import Trainer
from dp.utils import read_config, pickle_binary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for DeepPhonemizer.')
    parser.add_argument('--config', '-c', default='config.yaml', help='Points to the config file.')
    parser.add_argument('--checkpoint', '-cp', default=None, help='Points to the a model file to restore.')
    parser.add_argument('--path', '-p', help='Points to the a file with data.')
    args = parser.parse_args()

    config = read_config(args.config)

    if args.checkpoint:
        print(f'Restoring model from checkpoint: {args.checkpoint}')
        model, checkpoint = load_checkpoint(args.checkpoint)
        model.train()
        print(f'Loaded model with step: {model.get_step()}')
        for key, val in config['training'].items():
            val_orig = checkpoint['config']['training'][key]
            if val_orig != val:
                print(f'Overwriting training param: {key} {val_orig} --> {val}')
                checkpoint['config']['training'][key] = val
        config = checkpoint['config']
        model_type = config['model']['type']
    else:
        print('Initializing new model from config...')
        preprocessor = Preprocessor.from_config(config)
        model_type = config['model']['type']
        supported_types = ['lstm', 'transformer']
        if model_type == 'lstm':
            model = LstmModel.from_config(config)
        elif model_type == 'transformer':
            model = ForwardTransformer.from_config(config)
        elif model_type == 'autoreg_transformer':
            model = AutoregressiveTransformer.from_config(config)
        else:
            raise ValueError(f'Model type not supported: {model_type}. Supported types: {supported_types}')
        checkpoint = {
            'preprocessor': preprocessor,
            'config': config,
        }

    loss_type = 'cross_entropy' if 'autoreg_' in model_type else 'ctc'

    trainer = Trainer(checkpoint_dir=config['paths']['checkpoint_dir'], loss_type=loss_type)
    trainer.train(model=model,
                  checkpoint=checkpoint,
                  store_phoneme_dict_in_model=config['training']['store_phoneme_dict_in_model'])
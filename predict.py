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


if __name__ == '__main__':

    checkpoint_path = 'checkpoints/latest_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = TransformerModel.from_config(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()

    preprocessor = checkpoint['preprocessor']
    text_tok = preprocessor.text_tokenizer
    phon_tok = preprocessor.phoneme_tokenizer

    print(f'Restored model with step {model.get_step()}')

    text = 'Cov'

    tokens = text_tok(text)
    pred = model.generate(torch.tensor(tokens).unsqueeze(0))
    pred_decoded = phon_tok.decode(pred, remove_special_tokens=False)
    pred_decoded = ''.join(pred_decoded)
    print(f'{text} | {pred_decoded}')

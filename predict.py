import pickle
import argparse
import random
from collections import Counter
from typing import List, Tuple
import math
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

    text = 'Bleiben'

    tokens = text_tok(text)
    pred, logits = model.generate(torch.tensor(tokens).unsqueeze(0))
    norm_logits = logits.softmax(dim=2)
    probs = [norm_logits[0, i, p] for i, p in enumerate(pred[1:])]
    pred_decoded = phon_tok.decode(pred, remove_special_tokens=False)
    pred_decoded = ''.join(pred_decoded)
    prob = math.exp(sum([math.log(p) for p in probs]))
    for o, p in zip(pred_decoded[1:], probs):
        print(f'{o} {p}')
    print(f'{text} | {pred_decoded} | {prob}')


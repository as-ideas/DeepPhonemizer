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
from dp.predictor import Predictor
from dp.text import SequenceTokenizer
from dp.trainer import Trainer
from dp.utils import read_config


if __name__ == '__main__':

    checkpoint_path = 'checkpoints/latest_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = TransformerModel.from_config(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f'Restored model with step {model.get_step()}')

    preprocessor = checkpoint['preprocessor']

    predictor = Predictor(model=model, preprocessor=preprocessor)

    text = 'Bleiben'

    pred, meta = predictor([text], language='de')
    pred = pred[0]
    pred_decoded = ''.join(pred)

    tokens, logits = meta[0]['tokens'], meta[0]['logits']
    norm_logits = logits.softmax(dim=2)

    probs = [norm_logits[0, i, p] for i, p in enumerate(tokens[1:])]
    prob = math.exp(sum([math.log(p) for p in probs]))
    for o, p in zip(pred_decoded[1:], probs):
        print(f'{o} {p}')
    print(f'{text} | {pred_decoded} | {prob}')


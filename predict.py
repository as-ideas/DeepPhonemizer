import math
from itertools import groupby

import torch
from dp.predictor import Predictor
import math

import torch

from dp.predictor import Predictor

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/model_step_1700k.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    predictor = Predictor.from_checkpoint(checkpoint_path)

    text = ['sechshundert']

    pred_batch, meta = predictor(text, language='de')
    for i, pred in enumerate(pred_batch):
        pred_decoded = ''.join(pred)
        tokens, logits = meta[i]['tokens'], meta[i]['logits']

        norm_logits = logits.softmax(dim=1)
        probs = [norm_logits[i, p] for i, p in enumerate(tokens[1:])]

        prob = math.exp(sum([math.log(p) for p in probs]))


        for o, p in zip(pred_decoded, probs):
            print(f'{o} {p}')

        pred_decoded = [k for k, g in groupby(pred_decoded) if k != 0]
        pred_decoded = ''.join(pred_decoded)

        print(f'{text[i]} | {pred_decoded} | {prob}')


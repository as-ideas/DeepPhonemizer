import math
import torch
from dp.predictor import Predictor
import math

import torch

from dp.predictor import Predictor

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/latest_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    predictor = Predictor.from_checkpoint(checkpoint_path)

    text = ['hallo', 'bedeuten']

    pred_batch, meta = predictor(text, language='de')
    for i, pred in enumerate(pred_batch):
        pred_decoded = ''.join(pred)
        tokens, logits = meta[i]['tokens'], meta[i]['logits']
        norm_logits = logits.softmax(dim=1)
        probs = [norm_logits[i, p] for i, p in enumerate(tokens[1:])]
        prob = math.exp(sum([math.log(p) for p in probs]))

        for o, p in zip(pred_decoded[1:], probs):
            print(f'{o} {p}')
        print(f'{text[i]} | {pred_decoded} | {prob}')


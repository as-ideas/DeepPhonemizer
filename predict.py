import math
from itertools import groupby

import torch
from dp.predictor import Predictor
import math

import torch

from dp.predictor import Predictor

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/best_model_no_optim.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#    checkpoint['config']['model']['type'] = 'transformer'
#    torch.save(checkpoint, checkpoint_path)
    predictor = Predictor.from_checkpoint(checkpoint_path)

    text = ['hallo', 'du']

    pred_batch, meta = predictor(text, language='de', batch_size=1)
    tokens, probs = meta[0]['tokens'], meta[0]['probs']

    pred_decoded = predictor.phoneme_tokenizer.decode(
        tokens, remove_special_tokens=False)

    prob = math.exp(sum([math.log(p) for p in probs]))

    for o, p in zip(pred_decoded, probs):
        print(f'{o} {p}')

    #pred_decoded = [k for k, g in groupby(pred_decoded) if k != 0]
    pred_decoded = ''.join(pred_decoded)
    print(f'{text[0]} | {pred_decoded} | {prob}')


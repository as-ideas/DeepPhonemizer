import math
from itertools import groupby


import math

import torch

from dp.predictor import Predictor

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/best_model_no_optim.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print(f'model step {checkpoint["step"]}')
    predictor = Predictor.from_checkpoint(checkpoint_path)

    text = ['sicherlich', 'hallo']

    pred_batch, metas = predictor(text, language='de', batch_size=2)

    for i, meta in enumerate(metas):
        tokens, probs = meta['tokens'], meta['probs']

        pred_decoded = predictor.phoneme_tokenizer.decode(
            tokens, remove_special_tokens=False)

        prob = math.exp(sum([math.log(p) for p in probs]))

        for o, p in zip(pred_decoded, probs):
            print(f'{o} {p}')

        #pred_decoded = [k for k, g in groupby(pred_decoded) if k != 0]
        pred_decoded = ''.join(pred_decoded)
        print(f'{text[i]} {pred_decoded} | {prob}')


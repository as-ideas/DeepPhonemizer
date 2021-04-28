import math
from itertools import groupby


import math

import torch

from dp.predictor import Predictor

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/de_us_nostress/best_model_no_optim.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    predictor = Predictor.from_checkpoint(checkpoint_path)

    text = ['Engineering']

    predictions = predictor(text, lang='en_us', batch_size=2)

    for i, pred in enumerate(predictions):
        tokens, probs = pred.tokens, pred.token_probs

        pred_decoded = predictor.phoneme_tokenizer.decode(
            tokens, remove_special_tokens=False)

        prob = math.exp(sum([math.log(p) for p in probs]))

        for o, p in zip(pred_decoded, probs):
            print(f'{o} {p}')

        pred_decoded = ''.join(pred_decoded)
        print(f'{text[i]} {pred_decoded} | {prob}')


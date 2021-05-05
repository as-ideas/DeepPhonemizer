import logging

import torch
from dp.phonemizer import Phonemizer
from dp.preprocessing.text import Preprocessor

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/de_us_nostress_bind/best_model.pt'
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

    text = '(Schleswig-Holsteins)'

    result = phonemizer.phonemise_list([text], lang='de')
    print(result.phonemes)
    for text, pred in result.predictions.items():
        tokens, probs = pred.phoneme_tokens, pred.token_probs
        for o, p in zip(tokens, probs):
            print(f'{o} {p}')
        tokens = ''.join(tokens)
        print(f'{text} | {tokens} | {pred.confidence}')


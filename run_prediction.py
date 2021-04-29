import math
import torch
from dp.phonemizer import Phonemizer
from dp.preprocessing.text import Preprocessor

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/best_model_no_optim_onlymodel.pt'
    checkpoint = torch.load(checkpoint_path)
    checkpoint['preprocessor'] = Preprocessor.from_config(checkpoint['config'])
    torch.save(checkpoint, checkpoint_path)
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

    text = ['takatuka']

    result = phonemizer.phonemise_list(text, lang='de')

    for text, pred in result.predictions.items():
        tokens, probs = pred.tokens, pred.token_probs
        pred_decoded = phonemizer.predictor.phoneme_tokenizer.decode(
            tokens, remove_special_tokens=False)
        for o, p in zip(pred_decoded, probs):
            print(f'{o} {p}')
        pred_decoded = ''.join(pred_decoded)
        print(f'{text} | {pred_decoded} | {pred.confidence}')


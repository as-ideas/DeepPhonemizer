import math

from dp.phonemizer import Phonemizer

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/best_model_no_optim.pt'
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

    text = ['özdemir']

    result = phonemizer.phonemise_list(text, lang='de')

    for text, pred in result.predictions.items():
        tokens, probs = pred.tokens, pred.token_probs
        pred_decoded = phonemizer.predictor.phoneme_tokenizer.decode(
            tokens, remove_special_tokens=False)
        prob = math.exp(sum([math.log(p) for p in probs]))
        for o, p in zip(pred_decoded, probs):
            print(f'{o} {p}')
        pred_decoded = ''.join(pred_decoded)
        print(f'{text} {pred_decoded} | {prob}')


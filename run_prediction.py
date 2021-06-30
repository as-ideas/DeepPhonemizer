from dp.phonemizer import Phonemizer

if __name__ == '__main__':

    checkpoint_path = 'checkpoints/best_model_no_optim.pt'
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

    text = 'young'

    result = phonemizer.phonemise_list([text], lang='en_us')

    print(result.phonemes)
    for text, pred in result.predictions.items():
        tokens, probs = pred.phoneme_tokens, pred.token_probs
        for o, p in zip(tokens, probs):
            print(f'{o} {p}')
        tokens = ''.join(tokens)
        print(f'{text} | {tokens} | {pred.confidence}')


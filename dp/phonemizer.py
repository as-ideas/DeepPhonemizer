import torch
import re
from typing import Dict, Union, Tuple

from dp.model import TransformerModel


class Phonemizer:

    def __init__(self,
                 checkpoint_path: str,
                 lang_phoneme_dict: Dict[str, Dict[str, str]] = None) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model = TransformerModel.from_config(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model'])
        self.preprocessor = checkpoint['preprocessor']
        self.lang_phoneme_dict = lang_phoneme_dict

    def __call__(self,
                 text: str,
                 lang: str,
                 punctuation='().,:?!',
                 expand_acronyms=True) -> str:

        punctuation += ' '
        punc_set = set(punctuation + '-')
        punc_pattern = re.compile(f'([{punctuation}])')

        text = ''.join([t for t in text if t.isalnum() or t in punc_set])
        words = re.split(punc_pattern, text)

        # collect dictionary phonemes for words and hyphenated words
        word_phonemes = {word: self.get_dict_entry(word, lang, punc_set) for word in words}

        # collect dictionary phonemes for subwords in hyphenated words
        words_to_split = [w for w in words if word_phonemes[w] is None]
        word_splits = dict()
        for word in words_to_split:
            key = word
            if expand_acronyms:
                word = self.expand_acronym(word)
            word_split = re.split(r'([-])', word)
            word_splits[key] = word_split

        subwords = {w for values in word_splits.values() for w in values if len(w) > 0}
        for subword in subwords:
            if subword not in word_phonemes:
                word_phonemes[subword] = self.get_dict_entry(subword, lang, punc_set)

        # predict all non-hyphenated words and all subwords of hyphenated words
        words_to_predict = []
        for word, phons in word_phonemes.items():
            if phons is None and len(word_splits.get(word, [])) <= 1:
                words_to_predict.append(word)

        # can be batched
        for word in words_to_predict:
            phons = self.predict_word(word, lang)
            word_phonemes[word] = phons

        # collect all phonemes
        output = []
        for word in words:
            phons = word_phonemes[word]
            if phons is None:
                subwords = word_splits[word]
                subphons = [word_phonemes[w] for w in subwords]
                phons = ''.join(subphons)
            output.append(phons)
        return ''.join(output)

    def predict_word(self, word: str, lang: str) -> str:
        tokens = self.preprocessor.text_tokenizer(word)
        decoded = self.preprocessor.text_tokenizer.decode(tokens, remove_special_tokens=True)
        if len(decoded) == 0:
            return ''
        pred, _ = self.model.generate(torch.tensor(tokens).unsqueeze(0)).tolist()
        pred_decoded = self.preprocessor.phoneme_tokenizer.decode(pred, remove_special_tokens=True)
        phons = ''.join(pred_decoded)
        return phons

    def get_dict_entry(self,
                       word: str,
                       lang: str,
                       punc_set: set) -> Union[str, None]:
        if word in punc_set:
            return word
        if not self.lang_phoneme_dict or lang not in self.lang_phoneme_dict:
            return None
        phoneme_dict = self.lang_phoneme_dict[lang]
        if word in phoneme_dict:
            return phoneme_dict[word]
        elif word.lower() in phoneme_dict:
            return phoneme_dict[word.lower()]
        elif word.title() in phoneme_dict:
            return phoneme_dict[word.title()]
        else:
            return None

    def expand_acronym(self, word: str) -> str:
        subwords = []
        for subword in word.split('-'):
            if subword.isupper():
                subwords.append('-'.join(list(subword)))
            else:
                subwords.append(subword)
        return '-'.join(subwords)


if __name__ == '__main__':
    checkpoint_path = '../checkpoints/latest_model.pt'
    lang_phoneme_dict = {'de': {'E-Mail': 'ˈiːmeɪ̯l'}}
    phonemizer = Phonemizer(checkpoint_path=checkpoint_path,
                            lang_phoneme_dict=lang_phoneme_dict)
    phons = phonemizer('Der E-Mail kleine <SPD-Prinzen-könig - Francesco Cardinale, pillert an seinem Pillermann.', lang='de')
    print(phons)

import torch
import re
from typing import Dict, Union, Tuple, List

from dp.model import TransformerModel
from dp.predictor import Predictor
from dp.text import Preprocessor
from dp.utils import get_sequence_prob, load_checkpoint

DEFAULT_PUNCTUATION = '().,:?!'


class Phonemizer:

    def __init__(self,
                 predictor: Predictor,
                 preprocessor: Preprocessor,
                 lang_phoneme_dict: Dict[str, Dict[str, str]] = None) -> None:

        self.predictor = predictor
        self.preprocessor = preprocessor
        self.lang_phoneme_dict = lang_phoneme_dict

    def __call__(self,
                 text: Union[str, List[str]],
                 lang: str,
                 punctuation=DEFAULT_PUNCTUATION,
                 expand_acronyms=True) -> Union[str, List[str]]:
        """
        Phonemizes a single text or list of texts.

        :param texts: List texts to phonemize.
        :param lang: Language used for phonemization.
        :param punctuation: Punctuation symbols by which the texts are split.
        :param expand_acronyms: Whether to expand an acronym, e.g. DIY -> D-I-Y.
        :return: Phonemized texts.
        """

        single_input_string = isinstance(text, str)
        texts = [text] if single_input_string else text
        _, phoneme_lists, _ = self.phonemise_list(texts=texts, lang=lang,
                                        punctuation=punctuation, expand_acronyms=expand_acronyms)

        phoneme_lists = [''.join(phoneme_list) for phoneme_list in phoneme_lists]

        if single_input_string:
            return phoneme_lists[0]
        else:
            return phoneme_lists

    def phonemise_list(self,
                       texts: List[str],
                       lang: str,
                       punctuation=DEFAULT_PUNCTUATION,
                       expand_acronyms=True) -> Tuple[List[List[str]], List[List[str]],
                                                      Dict[str, Tuple[str, float]]]:

        """
        Phonemizes a list of texts and returns tokenized text and predictions with probabilities.

        :param texts: List texts to phonemize.
        :param lang: Language used for phonemization.
        :param punctuation: Punctuation symbols by which the texts are split.
        :param expand_acronyms: Whether to expand an acronym, e.g. DIY -> D-I-Y.
        :return: Tokenized input texts, phonemized input texts,
                 and model predictions as mapping of word to tuple (phonemes, probability).
        """

        punc_set = set(punctuation + '- ')
        punc_pattern = re.compile(f'([{punctuation + " "}])')

        cleaned_texts = []
        cleaned_words = set()
        for text in texts:
            cleaned_text = ''.join([t for t in text if t.isalnum() or t in punc_set])
            split = re.split(punc_pattern, cleaned_text)
            cleaned_texts.append(split)
            cleaned_words.update(split)

        # collect dictionary phonemes for words and hyphenated words
        word_phonemes = {word: self.get_dict_entry(word, lang, punc_set) for word in cleaned_words}

        # if words not in dictionary, try to split them into subwords (also keep non-splittable words)
        words_to_split = [w for w in cleaned_words if word_phonemes[w] is None]
        word_splits = dict()
        for word in words_to_split:
            key = word
            if expand_acronyms:
                word = self.expand_acronym(word)
            word_split = re.split(r'([-])', word)
            word_splits[key] = word_split

        # try to get dict entries of subwords (and whole words)
        subwords = {w for values in word_splits.values() for w in values if len(w) > 0}
        for subword in subwords:
            if subword not in word_phonemes:
                word_phonemes[subword] = self.get_dict_entry(subword, lang, punc_set)

        # predict all words and subwords that are missing in the phoneme dict
        words_to_predict = []
        for word, phons in word_phonemes.items():
            if phons is None and len(word_splits.get(word, [])) <= 1:
                words_to_predict.append(word)

        pred_phons, pred_probs = self.predict_words(words=words_to_predict, lang=lang)
        for word, phons in zip(words_to_predict, pred_phons):
            word_phonemes[word] = phons
        pred_word_probs = {word: (phon, prob) for word, phon, prob
                           in zip(words_to_predict, pred_phons, pred_probs) if len(word) > 0}

        # collect all phonemes
        output = []
        for text in cleaned_texts:
            out_phons = []
            for word in text:
                phons = word_phonemes[word]
                if phons is None:
                    subwords = word_splits[word]
                    subphons = [word_phonemes[w] for w in subwords]
                    phons = ''.join(subphons)
                out_phons.append(phons)
            output.append(out_phons)

        return cleaned_texts, output, pred_word_probs

    def predict_words(self, words: List[str], lang: str) -> Tuple[List[str], List[float]]:
        tokens, metas = self.predictor(words, language=lang)
        pred = [''.join(t) for t in tokens]
        probs = [get_sequence_prob(m['tokens'], m['logits']) for m in metas]
        return pred, probs

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

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: str,
                        device='cpu',
                        lang_phoneme_dict: Dict[str, Dict[str, str]] = None) -> 'Phonemizer':
        model, checkpoint = load_checkpoint(checkpoint_path, device=device)
        applied_phoneme_dict = None
        if lang_phoneme_dict is not None:
            applied_phoneme_dict = lang_phoneme_dict
        elif 'phoneme_dict' in checkpoint:
            applied_phoneme_dict = checkpoint['phoneme_dict']
        preprocessor = checkpoint['preprocessor']
        predictor = Predictor(model=model, preprocessor=preprocessor)
        return Phonemizer(predictor=predictor,
                          preprocessor=preprocessor,
                          lang_phoneme_dict=applied_phoneme_dict)


if __name__ == '__main__':
    checkpoint_path = '../checkpoints/best_model_no_optim.pt'
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

    input = 'YMCA die Waldfee.'
    words, phons, preds = phonemizer.phonemise_list([input], lang='de')
    print(words)
    print(phons)

    pred_words = sorted(list(preds.keys()), key=lambda x: -preds[x][1])
    for word in pred_words:
        pred, prob = preds[word]
        print(f'{word} {pred} {prob}')

    phons = [''.join(p) for p in phons]
    print(phons)

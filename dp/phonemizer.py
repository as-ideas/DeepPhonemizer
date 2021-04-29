import re
from itertools import zip_longest
from typing import Dict, Union, List, Set

from dp import PhonemizerResult
from dp.model.model import load_checkpoint
from dp.model.predictor import Predictor
from dp.preprocessing.text import Preprocessor


DEFAULT_PUNCTUATION = '().,:?!/'


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
                 expand_acronyms=True,
                 batch_size=8) -> Union[str, List[str]]:
        """
        Phonemizes a single text or list of texts.

        :param text: Text to phonemize as single string or list of strings.
        :param lang: Language used for phonemization.
        :param punctuation: Punctuation symbols by which the texts are split.
        :param expand_acronyms: Whether to expand an acronym, e.g. DIY -> D-I-Y.
        :param batch_size: Batch size of model to speed up inference.
        :return: Phonemized text as string, or list of strings, respectively.
        """

        single_input_string = isinstance(text, str)
        texts = [text] if single_input_string else text
        result = self.phonemise_list(texts=texts, lang=lang,
                                     punctuation=punctuation, expand_acronyms=expand_acronyms)

        phoneme_lists = [''.join(phoneme_list) for phoneme_list in result.phonemes]

        if single_input_string:
            return phoneme_lists[0]
        else:
            return phoneme_lists

    def phonemise_list(self,
                       texts: List[str],
                       lang: str,
                       punctuation=DEFAULT_PUNCTUATION,
                       expand_acronyms=True,
                       batch_size=8) -> PhonemizerResult:

        """
        Phonemizes a list of texts and returns tokenized texts, phonemes and word predictions with probabilities.

        :param texts: List texts to phonemize.
        :param lang: Language used for phonemization.
        :param punctuation: Punctuation symbols by which the texts are split.
        :param expand_acronyms: Whether to expand an acronym, e.g. DIY -> D-I-Y.
        :param batch_size: Batch size of model to speed up inference.
        :return: A tuple containing a nested list (tokenized input texts), a nested list (phonemized input texts),
                 and a dictionary (model predictions as mapping of word to tuple (phonemes, probability)).
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

        predictions = self.predictor(words=words_to_predict, lang=lang,
                                     batch_size=batch_size)
        for pred in predictions:
            word_phonemes[pred.word] = pred.phonemes
        pred_dict = {pred.word: pred for pred in predictions}

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

        return PhonemizerResult(text=cleaned_texts,
                                phonemes=output,
                                predictions=pred_dict)

    def get_dict_entry(self,
                       word: str,
                       lang: str,
                       punc_set: Set[str]) -> Union[str, None]:
        if word in punc_set or len(word) == 0:
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
            expanded = []
            for a, b in zip_longest(subword, subword[1:]):
                expanded.append(a)
                if b is not None and b.isupper():
                    expanded.append('-')
            expanded = ''.join(expanded)
            subwords.append(expanded)
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
    checkpoint_path = '../checkpoints/de_us_nostress/best_model_no_optim.pt'
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

    input = open('/Users/cschaefe/datasets/ASVoice4/metadata_clean_incl_english.csv').readlines()[-100:]
    input = [s.split('|')[1] for s in input if s.split('|')[0].startswith('en_') and len(s.split('|')) > 1][:]

    result = phonemizer.phonemise_list(input, lang='en_us', batch_size=8)
    for pred in sorted(result.predictions.values(), key=lambda p: -p.confidence):
        print(f'{pred.word} {pred.phonemes} {pred.confidence}')

    #print(result.phonemes)
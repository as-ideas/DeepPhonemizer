import re
from itertools import zip_longest
from typing import Dict, Union, List, Set

from dp import PhonemizerResult
from dp.model.model import load_checkpoint
from dp.model.predictor import Predictor
from dp.utils.logging import get_logger

DEFAULT_PUNCTUATION = '().,:?!/–'


class Phonemizer:

    def __init__(self,
                 predictor: Predictor,
                 lang_phoneme_dict: Dict[str, Dict[str, str]] = None) -> None:

        self.predictor = predictor
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
        Phonemizes a list of texts and returns tokenized texts,
        phonemes and word predictions with probabilities.

        :param texts: List texts to phonemize.
        :param lang: Language used for phonemization.
        :param punctuation: Punctuation symbols by which the texts are split.
        :param expand_acronyms: Whether to expand an acronym, e.g. DIY -> D-I-Y.
        :param batch_size: Batch size of model to speed up inference.
        :return: PhonemizerResult object containing original texts,
                 phonemes, split texts, split phonemes, and predictions
        """

        punc_set = set(punctuation + '- ')
        punc_pattern = re.compile(f'([{punctuation + " "}])')

        split_text, cleaned_words = [], set()
        for text in texts:
            cleaned_text = ''.join([t for t in text if t.isalnum() or t in punc_set])
            split = re.split(punc_pattern, cleaned_text)
            split = [s for s in split if len(s) > 0]
            split_text.append(split)
            cleaned_words.update(split)

        # collect dictionary phonemes for words and hyphenated words
        word_phonemes = {word: self.get_dict_entry(word=word, lang=lang, punc_set=punc_set)
                         for word in cleaned_words}

        # if word is not in dictionary, split it into subwords
        words_to_split = [w for w in cleaned_words if word_phonemes[w] is None]
        word_splits = dict()
        for word in words_to_split:
            key = word
            word = self.expand_acronym(word) if expand_acronyms else word
            word_split = re.split(r'([-])', word)
            word_splits[key] = word_split

        # collect dictionary entries of subwords
        subwords = {w for values in word_splits.values() for w in values}
        subwords = {w for w in subwords if w not in word_phonemes}
        for subword in subwords:
            word_phonemes[subword] = self.get_dict_entry(word=subword,
                                                         lang=lang,
                                                         punc_set=punc_set)

        # predict all subwords that are missing in the phoneme dict
        words_to_predict = [word for word, phons in word_phonemes.items()
                            if phons is None and len(word_splits.get(word, [])) <= 1]

        predictions = self.predictor(words=words_to_predict,
                                     lang=lang,
                                     batch_size=batch_size)

        word_phonemes.update({pred.word: pred.phonemes for pred in predictions})
        pred_dict = {pred.word: pred for pred in predictions}

        # collect all phonemes
        phoneme_lists = []
        for text in split_text:
            text_phons = [
                self._get_phonemes(word=word, word_phonemes=word_phonemes,
                                   word_splits=word_splits)
                for word in text
            ]
            phoneme_lists.append(text_phons)

        phonemes_joined = [''.join(phoneme_list) for phoneme_list in phoneme_lists]

        return PhonemizerResult(text=texts,
                                phonemes=phonemes_joined,
                                split_text=split_text,
                                split_phonemes=phoneme_lists,
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

    @staticmethod
    def expand_acronym(word: str) -> str:
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

    @staticmethod
    def _get_phonemes(word: str,
                      word_phonemes: Dict[str, Union[str, None]],
                      word_splits: Dict[str, List[str]]) -> str:
        phons = word_phonemes[word]
        if phons is None:
            subwords = word_splits[word]
            subphons = [word_phonemes[w] for w in subwords]
            phons = ''.join(subphons)
        return phons

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
        logger = get_logger(__name__)
        model_step = checkpoint['step']
        logger.debug(f'Initializing phonemizer with model step {model_step}')
        return Phonemizer(predictor=predictor,
                          lang_phoneme_dict=applied_phoneme_dict)
from collections import Counter

import tqdm
from typing import List, Iterable, Dict, Tuple


class Tokenizer:

    def __init__(self,
                 symbols: List[str],
                 lowercase=True,
                 pad_token='_',
                 start_token='<',
                 end_token='>') -> None:
        self.lowercase = lowercase
        self.pad_index = 0
        self.start_index = 1
        self.end_index = 2
        self.token_to_idx = {pad_token: self.pad_index,
                             start_token: self.start_index,
                             end_token: self.end_index}
        self.special_tokens = {pad_token, start_token, end_token}
        for symbol in symbols:
            self.token_to_idx[symbol] = len(self.token_to_idx)
        self.idx_to_token = {i: s for s, i in self.token_to_idx.items()}
        self.vocab_size = len(self.idx_to_token)

    def __call__(self, sentence: Iterable[str], append_start_end=True) -> List[int]:
        if self.lowercase:
            sentence = [s.lower() for s in sentence]
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        if append_start_end:
            sequence = [self.start_index] + sequence + [self.end_index]
        return sequence

    def decode(self, sequence: Iterable[int], remove_special_tokens=False) -> List[str]:
        decoded = [self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token]
        if remove_special_tokens:
            decoded = [d for d in decoded if d not in self.special_tokens]
        return decoded


class Preprocessor:

    def __init__(self,
                 lang_indices: Dict[str, int],
                 text_tokenizer: Tokenizer,
                 phoneme_tokenizer: Tokenizer) -> None:
        self.lang_indices = lang_indices
        self.text_tokenizer = text_tokenizer
        self.phoneme_tokenizer = phoneme_tokenizer

    def __call__(self,
                 data: List[Tuple[str, Iterable[str], Iterable[str]]],
                 progress=True) \
            -> List[Tuple[int, List[int], List[int]]]:
        data_processed = []
        data_iter = tqdm.tqdm(data, total=len(data)) if progress else iter(data)
        for lang, text, phonemes in data_iter:
            text_tokens = self.text_tokenizer(text)
            phoneme_tokens = self.phoneme_tokenizer(phonemes)
            lang_index = self.lang_indices[lang]
            data_processed.append((lang_index, text_tokens, phoneme_tokens))
        return data_processed

    @classmethod
    def build_from_data(cls, data: List[Tuple[str, Iterable[str], Iterable[str]]], lowercase=True) -> 'Preprocessor':
        lang_counter, text_counter, phoneme_counter = Counter(), Counter(), Counter()
        for lang, text, phonemes in data:
            lang_counter.update([lang])
            text_counter.update(text)
            phoneme_counter.update(phonemes)
        text_symbols = sorted((list(text_counter.keys())))
        phoneme_symbols = sorted(list(phoneme_counter.keys()))
        lang_symbols = sorted(list(lang_counter.keys()))
        lang_indices = {l: i for i, l in enumerate(lang_symbols)}
        text_tokenizer = Tokenizer(text_symbols, lowercase=lowercase)
        phoneme_tokenizer = Tokenizer(phoneme_symbols, lowercase=lowercase)
        return Preprocessor(lang_indices=lang_indices,
                            text_tokenizer=text_tokenizer,
                            phoneme_tokenizer=phoneme_tokenizer)

import shutil
from collections import Counter
from pathlib import Path
import torch
import tqdm
from typing import List, Iterable, Dict, Tuple, Any

from dp.utils import pickle_binary


class Tokenizer:

    def __init__(self,
                 symbols: List[str],
                 lowercase=False,
                 append_start_end=True,
                 pad_token='_',
                 start_token='<',
                 end_token='>') -> None:
        self.lowercase = lowercase
        self.append_start_end = append_start_end
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

    def __call__(self, sentence: Iterable[str]) -> List[int]:
        if self.lowercase:
            sentence = [s.lower() for s in sentence]
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        if self.append_start_end:
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
                 item: Tuple[str, Iterable[str], Iterable[str]]):

        lang, text, phonemes = item
        lang_index = self.lang_indices[lang]
        text_tokens = self.text_tokenizer(text)
        phoneme_tokens = self.phoneme_tokenizer(phonemes)
        return lang_index, text_tokens, phoneme_tokens

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Preprocessor':
        text_symbols = config['preprocessing']['text_symbols']
        phoneme_symbols = config['preprocessing']['phoneme_symbols']
        lang_symbols = config['preprocessing']['languages']
        lang_indices = {l: i for i, l in enumerate(lang_symbols)}
        text_tokenizer = Tokenizer(text_symbols,
                                   lowercase=config['preprocessing']['lowercase'],
                                   append_start_end=False)
        phoneme_tokenizer = Tokenizer(phoneme_symbols,
                                      lowercase=False,
                                      append_start_end=True)
        return Preprocessor(lang_indices=lang_indices,
                            text_tokenizer=text_tokenizer,
                            phoneme_tokenizer=phoneme_tokenizer)

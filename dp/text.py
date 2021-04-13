from typing import List, Iterable, Dict, Tuple, Any


class Tokenizer:

    def __init__(self,
                 symbols: List[str],
                 lowercase=False,
                 append_start_end=True,
                 start_index=1,
                 end_index=2,
                 pad_token='_',
                 start_token='<',
                 end_token='>') -> None:
        self.lowercase = lowercase
        self.append_start_end = append_start_end
        self.pad_index = 0
        self.start_index = start_index
        self.end_index = end_index
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
                 lang_tokenizer: Tokenizer,
                 text_tokenizer: Tokenizer,
                 phoneme_tokenizer: Tokenizer) -> None:
        self.lang_tokenizer = lang_tokenizer
        self.text_tokenizer = text_tokenizer
        self.phoneme_tokenizer = phoneme_tokenizer

    def __call__(self,
                 item: Tuple[str, Iterable[str], Iterable[str]]):

        lang, text, phonemes = item
        lang_token = self.lang_tokenizer([lang])[0]
        text_tokens = self.text_tokenizer(text)
        phoneme_tokens = self.phoneme_tokenizer(phonemes)
        return lang_token, text_tokens, phoneme_tokens

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Preprocessor':
        text_symbols = config['preprocessing']['text_symbols']
        phoneme_symbols = config['preprocessing']['phoneme_symbols']
        lang_symbols = config['preprocessing']['languages']
        start_index = config['preprocessing']['tokenizer_start_index']
        end_index = config['preprocessing']['tokenizer_end_index']
        lowercase = config['preprocessing']['lowercase']
        lang_tokenizer = Tokenizer(lang_symbols,
                                   lowercase=False,
                                   append_start_end=False)
        text_tokenizer = Tokenizer(text_symbols,
                                   lowercase=lowercase,
                                   start_index=start_index,
                                   end_index=end_index,
                                   append_start_end=False)
        phoneme_tokenizer = Tokenizer(phoneme_symbols,
                                      lowercase=False,
                                      start_index=start_index,
                                      end_index=end_index,
                                      append_start_end=True)
        return Preprocessor(lang_tokenizer=lang_tokenizer,
                            text_tokenizer=text_tokenizer,
                            phoneme_tokenizer=phoneme_tokenizer)

from typing import List, Iterable


class Tokenizer:

    def __init__(self,
                 symbols: List[str],
                 pad_token='_',
                 start_token='<',
                 end_token='>') -> None:
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
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        if append_start_end:
            sequence = [self.start_index] + sequence + [self.end_index]
        return sequence

    def decode(self, sequence: Iterable[int], remove_special_tokens=False) -> List[str]:
        decoded = [self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token]
        if remove_special_tokens:
            decoded = [d for d in decoded if d not in self.special_tokens]
        return decoded

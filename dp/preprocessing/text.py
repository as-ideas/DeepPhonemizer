from typing import List, Iterable, Dict, Tuple, Any


class LanguageTokenizer:

    """ Simple tokenizer for language to index mapping. """

    def __init__(self, languages: List[str]) -> None:
        """
        Initializes a language tokenizer for a list of languages.

        :param languages: List of languages, e.g. ['de', 'en'].
        """
        self.lang_index = {l: i for i, l in enumerate(languages)}
        self.index_lang = {i: l for i, l in enumerate(languages)}

    def __call__(self, lang: str) -> int:
        """
        Maps the language to an index.

        :param lang: Language to be mapped, e.g. 'de'.
        :return: Index of language.
        """

        if lang not in self.lang_index:
            raise ValueError(f'Language not supported: {lang}. '
                             f'Supported languages: {self.lang_index.keys()}')

        return self.lang_index[lang]

    def decode(self, index: int) -> str:
        """
        Inverts the index mapping of a language.

        :param index: Index of language.
        :return: Language as string.
        """
        return self.index_lang[index]


class SequenceTokenizer:

    """ Tokenizes text and attached language-specific start index (and non-specific end index). """

    def __init__(self,
                 symbols: List[str],
                 languages: List[str],
                 char_repeats: int,
                 lowercase=True,
                 append_start_end=True,
                 pad_token='_',
                 end_token='<end>') -> None:
        """
        Initializes a SequenceTokenizer object.

        :param symbols: Character (or phoneme) symbols.
        :param languages: List of languages.
        :param char_repeats: Number of repeats for each character to allow the forward model to map to longer
               phoneme sequences. Example: for char_repeats=2 the tokenizer maps hi -> hhii.
        :param lowercase: Whether to lowercase the input word.
        :param append_start_end: Whether to append special start and end tokens. Start and end tokens are
               index mappings of the chosen language.
        :param pad_token: Special pad token for index 0.
        :param end_token: Special end of sequence token.
        """

        self.languages = languages
        self.lowercase = lowercase
        self.char_repeats = char_repeats
        self.append_start_end = append_start_end
        self.pad_index = 0
        self.token_to_idx = {pad_token: self.pad_index}
        self.special_tokens = {pad_token, end_token}
        for lang in languages:
            lang_token = self._make_start_token(lang)
            self.token_to_idx[lang_token] = len(self.token_to_idx)
            self.special_tokens.add(lang_token)
        self.token_to_idx[end_token] = len(self.token_to_idx)
        self.end_index = self.token_to_idx[end_token]
        for symbol in symbols:
            self.token_to_idx[symbol] = len(self.token_to_idx)
        self.idx_to_token = {i: s for s, i in self.token_to_idx.items()}
        self.vocab_size = len(self.idx_to_token)

    def __call__(self, sentence: Iterable[str], language: str) -> List[int]:
        """
        Maps a sequence symbols for a language to a sequence of indices.

        :param sentence: Sentence (or word) as a sequence of symbols.
        :param language: Language for the mapping that defines the start and end token indices.
        :return: Sequence of token indices.
        """
        sentence = [item for item in sentence for i in range(self.char_repeats)]
        if language not in self.languages:
            raise ValueError(f'Language not supported: {language}. Supported languages: {self.languages}')
        if self.lowercase:
            sentence = [s.lower() for s in sentence]
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        if self.append_start_end:
            sequence = [self.get_start_index(language)] + sequence + [self.end_index]
        return sequence

    def decode(self, sequence: Iterable[int], remove_special_tokens=False) -> List[str]:
        """
        Inverts a sequence of indices to the original sequence of symbols.

        :param sequence: Encoded sequence to be decoded.
        :param remove_special_tokens: Whether to remove special tokens such as pad or start and end tokens.
        :return: Decoded sequence of symbols.
        """

        sequence = list(sequence)
        if self.append_start_end:
            sequence = sequence[:1] + sequence[1:-1:self.char_repeats] + sequence[-1:]
        else:
            sequence = sequence[::self.char_repeats]
        decoded = [self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token]
        if remove_special_tokens:
            decoded = [d for d in decoded if d not in self.special_tokens]
        return decoded

    def get_start_index(self, language: str) -> int:
        """
        Returns the start token index for a language.
        """
        lang_token = self._make_start_token(language)
        return self.token_to_idx[lang_token]

    def _make_start_token(self, language: str) -> str:
        return '<' + language + '>'


class Preprocessor:

    def __init__(self,
                 lang_tokenizer: LanguageTokenizer,
                 text_tokenizer: SequenceTokenizer,
                 phoneme_tokenizer: SequenceTokenizer) -> None:
        """
        Initializes a preprocessor object.

        :param lang_tokenizer: Tokenizer for input language.
        :param text_tokenizer: Tokenizer for input text.
        :param phoneme_tokenizer: Tokenizer for output phonemes.
        """

        self.lang_tokenizer = lang_tokenizer
        self.text_tokenizer = text_tokenizer
        self.phoneme_tokenizer = phoneme_tokenizer

    def __call__(self,
                 item: Tuple[str, Iterable[str], Iterable[str]]) -> Tuple[int, List[int], List[int]]:
        """
        Preprocesses a data tuple.
        :param item: Data point comprised of (language, input text, output phonemes).
        :return: Preprocessed data point as a tuple comprised of
                (language tokens, input_text tokens, output phonemes tokens)
        """

        lang, text, phonemes = item
        lang_token = self.lang_tokenizer(lang)
        text_tokens = self.text_tokenizer(text, lang)
        phoneme_tokens = self.phoneme_tokenizer(phonemes, lang)
        return lang_token, text_tokens, phoneme_tokens

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Preprocessor':
        """
        Initializes a preprocessor from a config.

        :param config: Dictionary containing preprocessing hyperparams.
        :return: Preprocessor object.
        """
        
        text_symbols = config['preprocessing']['text_symbols']
        phoneme_symbols = config['preprocessing']['phoneme_symbols']
        lang_symbols = config['preprocessing']['languages']
        char_repeats = config['preprocessing']['char_repeats']
        lowercase = config['preprocessing']['lowercase']
        lang_tokenizer = LanguageTokenizer(lang_symbols)
        text_tokenizer = SequenceTokenizer(symbols=text_symbols,
                                           languages=lang_symbols,
                                           char_repeats=char_repeats,
                                           lowercase=lowercase,
                                           append_start_end=True)
        phoneme_tokenizer = SequenceTokenizer(phoneme_symbols,
                                              languages=lang_symbols,
                                              lowercase=False,
                                              char_repeats=1,
                                              append_start_end=True)
        return Preprocessor(lang_tokenizer=lang_tokenizer,
                            text_tokenizer=text_tokenizer,
                            phoneme_tokenizer=phoneme_tokenizer)

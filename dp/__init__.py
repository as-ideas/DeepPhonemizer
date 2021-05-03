from typing import List, Dict


class Prediction:

    def __init__(self,
                 word: str,
                 phonemes: str,
                 phoneme_tokens: List[str],
                 confidence: float,
                 token_probs: List[float]) -> None:
        """
        Container for single word prediction.

        :param word: Original word to predict.
        :param phonemes: Predicted phonemes (without special tokens).
        :param phoneme_tokens: Predicted phoneme tokens (including special tokens).
        :param confidence: Total confidence of prediction.
        :param token_probs: Probability of each phoneme token.
        """

        self.word = word
        self.phonemes = phonemes
        self.phoneme_tokens = phoneme_tokens
        self.confidence = confidence
        self.token_probs = token_probs


class PhonemizerResult:

    def __init__(self,
                 text: List[str],
                 phonemes: List[str],
                 split_text: List[List[str]],
                 split_phonemes: List[List[str]],
                 predictions: Dict[str, Prediction]) -> None:
        """
        Container for phonemizer output.

        :param text: List of input texts
        :param phonemes: List of output phonemes
        :param split_text: List of texts, where each text is split into words and special chars
        :param split_phonemes: List of phonemes corresponding to split_text
        :param predictions: Dictionary with entries word to tuple (phoneme, probability)
        """

        self.text = text
        self.phonemes = phonemes
        self.split_text = split_text
        self.split_phonemes = split_phonemes
        self.predictions = predictions
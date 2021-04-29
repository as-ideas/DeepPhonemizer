from typing import List, Dict


class Prediction:

    def __init__(self,
                 word: str,
                 phonemes: str,
                 tokens: List[int],
                 confidence: float,
                 token_probs: List[float]) -> None:
        """
        Container for single word prediction.

        :param word: Original word to predict.
        :param phonemes: Predicted phonemes (without start and end token string).
        :param tokens: Predicted phoneme tokens (including start and end token).
        :param confidence: Total confidence of prediction.
        :param token_probs: Probability of each phoneme token.
        """

        self.word = word
        self.phonemes = phonemes
        self.tokens = tokens
        self.confidence = confidence
        self.token_probs = token_probs


class PhonemizerResult:

    def __init__(self,
                 text: List[List[str]],
                 phonemes: List[List[str]],
                 predictions: Dict[str, Prediction]) -> None:
        """
        Container for explicit phonemizer output.

        :param text: List of tokenized texts (list of words)
        :param phonemes: List of phonemes (list of word phonemes)
        :param predictions: Dictionary with entries word to Tuple (phoneme, probability)
        """

        self.text = text
        self.phonemes = phonemes
        self.predictions = predictions
import unittest
from typing import Tuple

import torch
from unittest.mock import Mock

from dp.predictor import Predictor
from dp.text import Preprocessor


def mock_model_call(input: torch.tensor) -> torch.tensor:

    """ returns input sandwiched with start and end indices """

    batch_size = input.size(0)
    logits = torch.full((batch_size, input.size(1), 20), fill_value=0.5)
    # return fake logits that result in output tokens = input tokens
    for t in range(input.size(1)):
        for b in range(input.size(0)):
            logits[b, t, input[b, t]] = 1.
    return logits


class TestPredictor(unittest.TestCase):

    def test_call_with_model_mock(self) -> None:
        model = Mock()
        model.side_effect = mock_model_call
        config = {
            'preprocessing': {
                'text_symbols': 'abcd',
                'phoneme_symbols': 'abcd',
                'char_repeats': 1,
                'languages': ['de'],
                'lowercase': False
            },
        }
        preprocessor = Preprocessor.from_config(config)
        predictor = Predictor(model, preprocessor)
        texts = ['ab', 'cde']

        phonemes, meta = predictor(texts, language='de', batch_size=8)
        self.assertEqual(2, len(phonemes))
        self.assertEqual(2, len(meta))
        self.assertEqual(['a', 'b'], phonemes[0])
        self.assertEqual(['c', 'd'], phonemes[1])

        phonemes, meta = predictor(texts, language='de', batch_size=1)
        self.assertEqual(2, len(phonemes))
        self.assertEqual(2, len(meta))
        self.assertEqual(['a', 'b'], phonemes[0])
        self.assertEqual(['c', 'd'], phonemes[1])

        texts = ['/']
        phonemes, meta = predictor(texts, language='de', batch_size=1)
        self.assertEqual(1, len(phonemes))
        self.assertEqual(1, len(meta))
        self.assertEqual([], phonemes[0])

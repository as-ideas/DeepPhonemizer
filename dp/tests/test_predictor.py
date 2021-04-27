import unittest
from typing import Dict, Any, Tuple
from unittest.mock import Mock

import torch

from dp.predictor import Predictor
from dp.text import Preprocessor


def mock_generate(batch: Dict[str, Any]) -> Tuple[torch.tensor, torch.tensor]:
    """ Return input and ones as probs """
    tokens = batch['text']
    probs = torch.ones(tokens.size())
    return tokens, probs


class TestPredictor(unittest.TestCase):

    def test_call_with_model_mock(self) -> None:
        model = Mock()
        model.generate = mock_generate
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

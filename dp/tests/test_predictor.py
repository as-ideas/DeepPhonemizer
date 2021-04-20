import unittest
from typing import Tuple

import torch
from unittest.mock import Mock

from dp.predictor import Predictor
from dp.text import Preprocessor


def mock_generate(input: torch.tensor,
                  start_index: int,
                  end_index: int) -> Tuple[torch.tensor, torch.tensor]:

    """ returns input sandwiched with start and end indices """

    batch_size = input.size(0)
    start_tens = torch.full((batch_size, 1), fill_value=start_index)
    end_tens = torch.full((batch_size, 1), fill_value=end_index)
    tokens = torch.cat([start_tens, input, end_tens], dim=1)
    logits = torch.full((batch_size, tokens.size(1), 20), fill_value=0.5)
    return tokens, logits


class TestPredictor(unittest.TestCase):

    def test_call_with_model_mock(self) -> None:
        model = Mock()
        model.generate = mock_generate
        config = {
            'preprocessing': {
                'text_symbols': 'abcd',
                'phoneme_symbols': 'abcd',
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
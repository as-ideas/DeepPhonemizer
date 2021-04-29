import unittest
from typing import List
from unittest.mock import patch

from dp.phonemizer import Phonemizer
from dp.model.predictor import Predictor, Prediction
from dp.preprocessing.text import Preprocessor


class PredictorMock:

    def __call__(self, words: List[str], lang: str, **kwargs) -> List[Prediction]:
        output = []
        for word in words:
            phons = word + f'-phon-{lang}'
            pred = Prediction(word=word, phonemes=phons, confidence=0.5,
                              tokens=[1]*len(phons), token_probs=[0.5]*len(phons))
            output.append(pred)
        return output


class TestPhonemizer(unittest.TestCase):

    @patch.object(Predictor, '__call__', new_callable=PredictorMock)
    def test_call_with_predictor_mock(self, predictor: Predictor) -> None:

        config = {'preprocessing': {}}
        config['preprocessing']['text_symbols'] = 'abcdefghijklmnopqrstuvwxyz'
        config['preprocessing']['phoneme_symbols'] = 'abcdefghijklmnopqrstuvwxyz'
        config['preprocessing']['languages'] = ['de']
        config['preprocessing']['char_repeats'] = 1
        config['preprocessing']['lowercase'] = True
        preprocessor = Preprocessor.from_config(config)
        phonemizer = Phonemizer(predictor=predictor, preprocessor=preprocessor)

        result = phonemizer('hallo', lang='de')
        self.assertEqual('hallo-phon-de', result)

        result = phonemizer(['hallo', 'du'], lang='de')
        self.assertEqual(['hallo-phon-de', 'du-phon-de'], result)


        print(result)
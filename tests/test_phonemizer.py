import unittest
from typing import List
from unittest.mock import patch

from dp.phonemizer import Phonemizer
from dp.model.predictor import Predictor, Prediction


class PredictorMock:

    def __call__(self, words: List[str], lang: str, **kwargs) -> List[Prediction]:
        """ Simply returns the original words with a suffix -phon-{language} to test for calls. """

        output = []
        for word in words:
            phons = word + f'-phon-{lang}'
            pred = Prediction(word=word, phonemes=phons, confidence=0.5,
                              phoneme_tokens=list(phons), token_probs=[0.5]*len(phons))
            output.append(pred)
        return output


class TestPhonemizer(unittest.TestCase):

    @patch.object(Predictor, '__call__', new_callable=PredictorMock)
    def test_call_with_predictor_mock(self, predictor: Predictor) -> None:
        phonemizer = Phonemizer(predictor=predictor)

        result = phonemizer('hallo', lang='de')
        self.assertEqual('hallo-phon-de', result)

        result = phonemizer(['hallo', 'du'], lang='de')
        self.assertEqual(['hallo-phon-de', 'du-phon-de'], result)

    @patch.object(Predictor, '__call__', new_callable=PredictorMock)
    def test_apply_dictionary(self, predictor: Predictor) -> None:
        phoneme_dict = {'de': {}, 'en': {}}
        phoneme_dict['de']['die'] = 'DIE'
        phoneme_dict['de']['das'] = 'DAS'
        phoneme_dict['de']['das-die'] = 'DAS!DIE'
        phoneme_dict['en']['die'] = 'DAI'

        phonemizer = Phonemizer(predictor=predictor,
                                lang_phoneme_dict=phoneme_dict)

        result = phonemizer('die', lang='de')
        self.assertEqual('DIE', result)

        result = phonemizer('Die', lang='de')
        self.assertEqual('DIE', result)

        result = phonemizer('DIE', lang='de')
        self.assertEqual('DIE', result)

        # test whether the dict check gets hyphenated words
        result = phonemizer('das-die', lang='de')
        self.assertEqual('DAS!DIE', result)

        # test whether the dict check is applied to subwords in a hyphenated word
        result = phonemizer('die-das', lang='de')
        self.assertEqual('DIE-DAS', result)

        result = phonemizer('/die/?!', lang='de', punctuation='')
        self.assertEqual('DIE', result)

        result = phonemizer('/die/?!', lang='de', punctuation='/')
        self.assertEqual('/DIE/', result)

        result = phonemizer('die', lang='en')
        self.assertEqual('DAI', result)

        result = phonemizer('die-die-das', lang='en')
        self.assertEqual('DAI-DAI-das-phon-en', result)

        result = phonemizer('dies', lang='de')
        self.assertEqual('dies-phon-de', result)

        result = phonemizer('dies', lang='en')
        self.assertEqual('dies-phon-en', result)

    @patch.object(Predictor, '__call__', new_callable=PredictorMock)
    def test_split_result(self, predictor: Predictor) -> None:
        phonemizer = Phonemizer(predictor=predictor)

        text = 'Ich, bin - so wie (ich/du) bin.'
        result = phonemizer.phonemise_list([text], lang='de', punctuation=',().')
        cleaned_text = ''.join(result.split_text[0])
        self.assertEqual('Ich, bin - so wie (ichdu) bin.', cleaned_text)
        self.assertEqual(len(result.split_text[0]), len(result.split_phonemes[0]))

        text = 'Ich, bin: - (so.'
        result = phonemizer.phonemise_list([text], lang='de', punctuation=',().')
        self.assertEqual(['Ich', ',', ' ', 'bin', ' ', '-', ' ', '(', 'so', '.'],
                         result.split_text[0])

        # test some boundaries
        result = phonemizer('whitespace ', lang='de')
        self.assertEqual('whitespace-phon-de ', result)

        # test some boundaries
        result = phonemizer(',special!?', lang='de', punctuation=',!')
        self.assertEqual(',special-phon-de!', result)

        # test some boundaries
        result = phonemizer(',special!?', lang='de', punctuation='')
        self.assertEqual('special-phon-de', result)

    def test_expand_acronym(self) -> None:
        expanded = Phonemizer._expand_acronym('hallo')
        self.assertEqual('hallo', expanded)

        expanded = Phonemizer._expand_acronym('HAL')
        self.assertEqual('H-A-L', expanded)

        expanded = Phonemizer._expand_acronym('hallO')
        self.assertEqual('hall-O', expanded)

        expanded = Phonemizer._expand_acronym('HaLO')
        self.assertEqual('Ha-L-O', expanded)

        expanded = Phonemizer._expand_acronym('vCPU')
        self.assertEqual('v-C-P-U', expanded)

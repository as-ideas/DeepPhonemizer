import unittest

from dp.training.evaluation import evaluate_samples


class TestEvaluation(unittest.TestCase):

    def test_evaluate_samples(self):

        lang_samples = {
            'de': [
                (['a'], ['a', 'b'], ['a', 'b']),
            ],
            'en_us': [
                # two alternative targets, evaluation should take the closer one with per=1./3
                (['a'], ['a', 'b', 'c'], ['b', 'b', 'b']),
                (['a'], ['a', 'b', 'c'], ['a', 'b', 'b'])
            ]
        }

        result = evaluate_samples(lang_samples=lang_samples)

        self.assertEqual(0., result['de']['per'])
        self.assertEqual(0., result['de']['wer'])
        self.assertEqual(1./3, result['en_us']['per'])
        self.assertEqual(1., result['en_us']['wer'])
        self.assertEqual((0 + 1) / (2 + 3), result['mean_per'])
        self.assertEqual((0 + 1) / 2, result['mean_wer'])
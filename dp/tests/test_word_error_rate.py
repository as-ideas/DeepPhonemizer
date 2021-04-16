import unittest

from dp.metrics import word_error_rate


class TestWordErrorRate(unittest.TestCase):

    def test_call(self):
        predicted = ['a', 'b', 'c', 'd']
        target = ['a', 'k', 'c', 'a']
        wer = word_error_rate(predicted, target)
        self.assertEqual(1, wer)

        predicted = ['r', 'r', 'r', 'r']
        target = ['a', 'k', 'c', 'a']
        wer = word_error_rate(predicted, target)
        self.assertEqual(1., wer)

        predicted = ['a']
        target = ['a']
        wer = word_error_rate(predicted, target)
        self.assertEqual(0, wer)

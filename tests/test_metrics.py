import unittest

from dp.training.metrics import word_error, phoneme_error_rate


class TestWordError(unittest.TestCase):

    def test_call(self):
        predicted = ['a', 'b', 'c', 'd']
        target = ['a', 'k', 'c', 'a']
        result = word_error(predicted, target)
        self.assertEqual(1, result)

        predicted = ['r', 'r', 'r', 'r']
        target = ['a', 'k', 'c', 'a']
        result = word_error(predicted, target)
        self.assertEqual(1., result)

        predicted = ['a']
        target = ['a']
        result = word_error(predicted, target)
        self.assertEqual(0, result)


class TestPhonemeErrorRate(unittest.TestCase):

    def test_call(self):
        predicted = ['a', 'b', 'c', 'd']
        target = ['a', 'k', 'c', 'a']
        result = phoneme_error_rate(predicted, target)
        self.assertEqual(0.5, result)

        predicted = ['r', 'r', 'r', 'r']
        target = ['a', 'k', 'c', 'a']
        result = phoneme_error_rate(predicted, target)
        self.assertEqual(1., result)

        predicted = ['a']
        target = ['a']
        result = phoneme_error_rate(predicted, target)
        self.assertEqual(0, result)

        predicted = ['a']
        target = ['b']
        result = phoneme_error_rate(predicted, target)
        self.assertEqual(1., result)

        predicted = ['a', 'b']
        target = ['b']
        result = phoneme_error_rate(predicted, target)
        self.assertEqual(1., result)

        predicted = ['a', 'b', 'c']
        target = ['b']
        result = phoneme_error_rate(predicted, target)
        self.assertEqual(2., result)

        predicted = ['a']
        target = ['a', 'b']
        result = phoneme_error_rate(predicted, target)
        self.assertEqual(0.5, result)

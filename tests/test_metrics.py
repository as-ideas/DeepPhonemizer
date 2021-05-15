import unittest

from dp.training.metrics import word_error, phoneme_error


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
        e, c = phoneme_error(predicted, target)
        self.assertEqual(2, e)
        self.assertEqual(4, c)

        predicted = ['r', 'r', 'r', 'r']
        target = ['a', 'k', 'c', 'a']
        e, c = phoneme_error(predicted, target)
        self.assertEqual(4, e)
        self.assertEqual(4, c)

        predicted = ['a']
        target = ['a']
        e, c = phoneme_error(predicted, target)
        self.assertEqual(0, e)
        self.assertEqual(1, c)

        predicted = ['a']
        target = ['b']
        e, c = phoneme_error(predicted, target)
        self.assertEqual(1, e)
        self.assertEqual(1, c)

        predicted = ['a', 'b']
        target = ['b']
        e, c = phoneme_error(predicted, target)
        self.assertEqual(1, e)
        self.assertEqual(1, c)

        predicted = ['a', 'b', 'c']
        target = ['b']
        e, c = phoneme_error(predicted, target)
        self.assertEqual(2, e)
        self.assertEqual(1, c)

        predicted = ['a']
        target = ['a', 'b']
        e, c = phoneme_error(predicted, target)
        self.assertEqual(1, e)
        self.assertEqual(2, c)

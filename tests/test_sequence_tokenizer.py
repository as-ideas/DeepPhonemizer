import unittest

from dp.preprocessing.text import SequenceTokenizer


class TestSequenceTokenizer(unittest.TestCase):

    def test_call_happy_path(self) -> None:
        symbols = ['a', 'b', 'c', 'd', 'e']
        languages = ['de', 'en']

        tokenizer = SequenceTokenizer(symbols=symbols, languages=languages, char_repeats=1,
                                      lowercase=True, append_start_end=True, end_token='<end>')

        tokens = tokenizer(['a', 'b'], language='de')
        decoded = tokenizer.decode(tokens)

        self.assertEqual(4, len(tokens))
        self.assertEqual(['<de>', 'a', 'b', '<end>'], decoded)

        tokens = tokenizer(['a', 'b'], language='en')
        decoded = tokenizer.decode(tokens)

        self.assertEqual(4, len(tokens))
        self.assertEqual(['<en>', 'a', 'b', '<end>'], decoded)

    def test_call_missing_symbols(self) -> None:
        symbols = ['a', 'b']
        languages = ['de', 'en']

        tokenizer = SequenceTokenizer(symbols=symbols, languages=languages,  lowercase=False,
                                      append_start_end=True, end_token='<end>', char_repeats=2)

        tokens = tokenizer(['A', 'b', 'F'], language='en')
        decoded = tokenizer.decode(tokens)

        self.assertEqual(4, len(tokens))
        self.assertEqual(['<en>', 'b', '<end>'], decoded)

    def test_call_edge_cases(self) -> None:
        symbols = ['a', 'b', 'c', 'd', 'e']
        languages = ['de', 'en']

        tokenizer = SequenceTokenizer(symbols=symbols, languages=languages,  lowercase=True,
                                      append_start_end=True, end_token='<end>', char_repeats=2)

        tokens = tokenizer([], language='en')
        decoded = tokenizer.decode(tokens)

        self.assertEqual(2, len(tokens))
        self.assertEqual(['<en>', '<end>'], decoded)

        tokens = tokenizer(['z'], language='en')
        decoded = tokenizer.decode(tokens)

        self.assertEqual(2, len(tokens))
        self.assertEqual(['<en>', '<end>'], decoded)

        tokenizer = SequenceTokenizer(symbols=['a'], languages=['de'],  lowercase=True,
                                      append_start_end=False, end_token=None, char_repeats=3)

        tokens = tokenizer(['a'], language='de')
        decoded = tokenizer.decode(tokens)

        self.assertEqual(3, len(tokens))
        self.assertEqual(['a'], decoded)

    def test_exception(self) -> None:
        tokenizer = SequenceTokenizer(symbols=['a'], languages=['de'],  lowercase=True,
                                      append_start_end=True, end_token='<end>', char_repeats=2)

        with self.assertRaises(ValueError):
            tokenizer(['a', 'b'], language='not_existent')


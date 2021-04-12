import unittest

from dp.text import Tokenizer


class TestTokenizer(unittest.TestCase):

    def test_tokenizer(self) -> None:
        symbols = list('abcde')
        tokenizer = Tokenizer(symbols=symbols, lowercase=True, start_index=1,  end_index=2,
                              start_token='<', end_token='>', append_start_end=True)

        tokens = tokenizer('aBk')
        self.assertEqual([1, 3, 4, 2], tokens)

        decoded = tokenizer.decode(tokens)
        self.assertEqual(['<', 'a', 'b', '>'], decoded)

        decoded = tokenizer.decode(tokens, remove_special_tokens=True)
        self.assertEqual(['a', 'b'], decoded)

        symbols = list('abcde')
        tokenizer = Tokenizer(symbols=symbols, lowercase=False, start_index=1,  end_index=2,
                              start_token='<', end_token='>', append_start_end=True)

        tokens = tokenizer('aBk')
        self.assertEqual([1, 3, 2], tokens)

        decoded = tokenizer.decode(tokens, remove_special_tokens=True)
        self.assertEqual(['a'], decoded)

        tokenizer = Tokenizer(symbols=symbols, lowercase=True, start_index=1,  end_index=2,
                              start_token='<', end_token='>', append_start_end=False)

        tokens = tokenizer('aBk')
        self.assertEqual([3, 4], tokens)

        decoded = tokenizer.decode(tokens, remove_special_tokens=False)
        self.assertEqual(['a', 'b'], decoded)

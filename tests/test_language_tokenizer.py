import unittest

from dp.preprocessing.text import LanguageTokenizer


class TestLanguageTokenizer(unittest.TestCase):

    def test_call_happy_path(self) -> None:

        tokenizer = LanguageTokenizer(languages=['de', 'en'])

        de_tok = tokenizer('de')
        en_tok = tokenizer('en')
        self.assertEqual(0, de_tok)
        self.assertEqual(1, en_tok)

        self.assertEqual('de', tokenizer.decode(de_tok))
        self.assertEqual('en', tokenizer.decode(en_tok))

    def test_exception(self) -> None:
        tokenizer = LanguageTokenizer(languages=['de', 'en'])

        with self.assertRaises(ValueError):
            tokenizer('not_existent')


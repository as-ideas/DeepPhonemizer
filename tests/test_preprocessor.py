import unittest

from dp.preprocessing.text import Preprocessor


class TestPreprocessor(unittest.TestCase):

    def test_call_happy_path(self) -> None:
        config = {
            'preprocessing': {
                'text_symbols': 'abcdABCD',
                'phoneme_symbols': 'abcd',
                'char_repeats': 1,
                'languages': ['de', 'en'],
                'lowercase': False
            },
        }

        preprocessor = Preprocessor.from_config(config)
        result = preprocessor(('de', 'aB', 'cd'))
        lang, text, phons = result
        lang = preprocessor.lang_tokenizer.decode(lang)
        text = preprocessor.text_tokenizer.decode(text, remove_special_tokens=False)
        phons = preprocessor.phoneme_tokenizer.decode(phons, remove_special_tokens=False)
        self.assertEqual('de', lang)
        self.assertEqual(['<de>', 'a', 'B', '<end>'], text)
        self.assertEqual(['<de>', 'c', 'd', '<end>'], phons)
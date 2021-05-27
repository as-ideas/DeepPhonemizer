import os
import shutil
import tempfile
import unittest
from pathlib import Path

from dp import preprocess
from dp.preprocess import preprocess
from dp.preprocessing.text import Preprocessor
from dp.utils.io import read_config, unpickle_binary, save_config


class TestPreprocess(unittest.TestCase):

    def setUp(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix='TestPreprocessTmp')
        self.temp_dir = Path(temp_dir)
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.test_config_path = Path(test_path) / 'resources/forward_test_config.yaml'

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preprocess_happy_path(self) -> None:
        config = read_config(self.test_config_path)
        data_dir = self.temp_dir / 'datasets'
        config_path = self.temp_dir / 'forward_test_config.yaml'
        config['paths']['data_dir'] = str(data_dir)
        save_config(config, config_path)

        train_data = [
            ('de','benützten', 'bəˈnʏt͡stn̩'),
            ('de', 'gewürz', 'ɡəˈvʏʁt͡s'),
            ('en_us', 'young', 'jʌŋ'),
        ]

        val_data = [
            ('en_us', 'young', 'jʌŋ'),
            ('de', 'gewürz', 'ɡəˈvʏʁt͡s')
        ]

        preprocess(config_file=config_path,
                   train_data=train_data,
                   val_data=val_data,
                   deduplicate_train_data=False)

        preprocessor = Preprocessor.from_config(config)

        expected_train = [preprocessor(t) for t in train_data]
        actual_train = unpickle_binary(data_dir / 'train_dataset.pkl')
        self.assertEqual(expected_train, actual_train)

        expected_val = [preprocessor(t) for t in val_data]
        actual_val = unpickle_binary(data_dir / 'val_dataset.pkl')
        self.assertEqual(expected_val, actual_val)

        expected_phon_dict = {'de': {}, 'en_us': {}}
        for lang, word, phon in train_data + val_data:
            expected_phon_dict[lang][word] = phon

        actual_phon_dict = unpickle_binary(data_dir / 'phoneme_dict.pkl')
        self.assertEqual(expected_phon_dict, actual_phon_dict)
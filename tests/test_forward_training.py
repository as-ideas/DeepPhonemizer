import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dp import preprocess
from dp.model.model import ForwardTransformer
from dp.model.predictor import Predictor
from dp.preprocess import preprocess
from dp.train import train
from dp.utils.io import read_config, save_config


class TestForwardTraining(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        temp_dir = tempfile.mkdtemp(prefix='TestPreprocessTmp')
        self.temp_dir = Path(temp_dir)
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.test_config_path = Path(test_path) / 'resources/forward_test_config.yaml'

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_forward_training_happy_path(self) -> None:
        config = read_config(self.test_config_path)
        data_dir = self.temp_dir / 'datasets'
        checkpoint_dir = self.temp_dir / 'checkpoints'
        config_path = self.temp_dir / 'forward_test_config.yaml'
        config['paths']['data_dir'] = str(data_dir)
        config['paths']['checkpoint_dir'] = str(checkpoint_dir)
        save_config(config, config_path)

        train_data = [
            ('en_us', 'young', 'jʌŋ'),
            ('de', 'benützten', 'bənʏt͡stn̩'),
            ('de', 'gewürz', 'ɡəvʏʁt͡s')
        ] * 100

        val_data = [
            ('en_us', 'young', 'jʌŋ'),
            ('de', 'gewürz', 'ɡəvʏʁt͡s')
        ] * 10

        preprocess(config_file=config_path,
                   train_data=train_data,
                   val_data=val_data,
                   deduplicate_train_data=False)

        train(rank=0, num_gpus=0, config_file=config_path)

        predictor = Predictor.from_checkpoint(checkpoint_dir / 'latest_model.pt')

        self.assertTrue(isinstance(predictor.model, ForwardTransformer))

        result = predictor(words=['young'], lang='en_us')[0]
        self.assertEqual('jʌŋ', result.phonemes)
        self.assertTrue(result.confidence > 0.95)

        result = predictor(words=['gewürz'], lang='de')[0]
        self.assertEqual('ɡəvʏʁt͡s', result.phonemes)
        self.assertTrue(result.confidence > 0.93)

        result = predictor(words=['gewürz'], lang='en_us')[0]
        self.assertEqual('ɡəvʏʁt͡s', result.phonemes)
        self.assertTrue(0.82 < result.confidence < 0.86)

        # test jit export
        predictor.model = torch.jit.script(predictor.model)
        result_jit = predictor(words=['gewürz'], lang='en_us')[0]
        self.assertEqual(result.phonemes, result_jit.phonemes)
        self.assertEqual(result.confidence, result_jit.confidence)

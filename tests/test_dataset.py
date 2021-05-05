import unittest
import torch
from dp.training.dataset import PhonemizerDataset, collate_dataset


class TestPhonemizerDataset(unittest.TestCase):

    def test_get_items(self) -> None:
        items_input = [
            (0, [1, 2], [3, 4]),
            (1, [2, 3], [4, 5, 6])
        ]
        items = PhonemizerDataset(items=items_input)

        self.assertEqual(0, items[0]['item_id'])
        self.assertEqual(0, items[0]['language'])
        self.assertEqual([1, 2], items[0]['text'].tolist())
        self.assertEqual([3, 4], items[0]['phonemes'].tolist())
        self.assertEqual(2, items[0]['text_len'])
        self.assertEqual(2, items[0]['phonemes_len'])
        self.assertEqual(3, items[0]['start_index'])

        self.assertEqual(1, items[1]['item_id'])
        self.assertEqual(1, items[1]['language'])

    def test_collate(self):
        batch = [
            {
                'item_id': 0, 'language': 0, 'text_len': 2,
                'phonemes_len': 2, 'start_index': 2,
                'text': torch.tensor([1, 2]),
                'phonemes': torch.tensor([2, 3])
            },
            {
                'item_id': 1, 'language': 1, 'text_len': 3,
                'phonemes_len': 3, 'start_index': 3,
                'text': torch.tensor([1, 2, 3]),
                'phonemes': torch.tensor([3, 4, 5])
            }
        ]

        batch = collate_dataset(batch)
        self.assertEqual(0, batch['item_id'][0])
        self.assertEqual(1, batch['item_id'][1])
        self.assertEqual(2, batch['text_len'][0])
        self.assertEqual(3, batch['text_len'][1])
        self.assertEqual(2, batch['phonemes_len'][0])
        self.assertEqual(3, batch['phonemes_len'][1])
        self.assertEqual(0, batch['language'][0])
        self.assertEqual(1, batch['language'][1])
        self.assertEqual([1, 2, 0], batch['text'][0].tolist())
        self.assertEqual([1, 2, 3], batch['text'][1].tolist())
        self.assertEqual([2, 3, 0], batch['phonemes'][0].tolist())
        self.assertEqual([3, 4, 5], batch['phonemes'][1].tolist())

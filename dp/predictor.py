from typing import Dict, Any, List, Tuple, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence

from dp.model import load_checkpoint
from dp.model_utils import get_len_util_stop
from dp.text import Preprocessor
from dp.utils import batchify


class Predictor:

    def __init__(self,
                 model: torch.nn.Module,
                 preprocessor: Preprocessor) -> None:
        self.model = model
        self.text_tokenizer = preprocessor.text_tokenizer
        self.phoneme_tokenizer = preprocessor.phoneme_tokenizer

    def __call__(self,
                 words: List[str],
                 language: str,
                 batch_size=8) -> Tuple[List[Iterable[str]], List[Dict[str, Any]]]:
        """
        Predicts phonemes for a list of words.

        :param words: List of words to predict.
        :param language: Language of texts.
        :param batch_size: Size of batch for model input to speed up inference.
        :return: A tuple containing a list (predicted phonemes)
                 and a list (additional info per prediction such as logits, probability etc.)
        """

        predictions = dict()
        valid_texts = set()

        # handle words that result in an empty input to the model
        for word in words:
            input = self.text_tokenizer(sentence=word, language=language)
            decoded = self.text_tokenizer.decode(
                sequence=input, remove_special_tokens=True)
            if len(decoded) == 0:
                predictions[word] = ([], None)
            else:
                valid_texts.add(word)

        valid_texts = sorted(list(valid_texts), key=lambda x: len(x))
        batch_pred = self._predict_batch(texts=valid_texts, batch_size=batch_size,
                                         language=language)
        predictions.update(batch_pred)

        out_phonemes, out_meta = [], []
        for word in words:
            output, probs = predictions[word]
            out_phons = self.phoneme_tokenizer.decode(
                sequence=output, remove_special_tokens=True)
            out_phonemes.append(out_phons)
            out_meta.append({'phonemes': out_phons, 'probs': probs, 'tokens': output})

        return out_phonemes, out_meta

    def _predict_batch(self,
                       texts: List[str],
                       batch_size: int,
                       language: str) \
            -> Dict[str, Tuple[torch.tensor, torch.tensor]]:
        predictions = dict()
        text_batches = batchify(texts, batch_size)
        for text_batch in text_batches:
            input_batch, lens_batch = [], []
            for text in text_batch:
                input = self.text_tokenizer(text, language)
                input_batch.append(torch.tensor(input).long())
                lens_batch.append(torch.tensor(len(input)))

            input_batch = pad_sequence(sequences=input_batch,
                                       batch_first=True, padding_value=0)
            lens_batch = torch.stack(lens_batch)
            start_indx = self.phoneme_tokenizer.get_start_index(language)
            start_inds = torch.tensor([start_indx]*input_batch.size(0)).to(input_batch.device)
            batch = {
                'text': input_batch,
                'text_len': lens_batch,
                'start_index': start_inds
            }
            with torch.no_grad():
                output_batch, probs_batch = self.model.generate(batch)
            output_batch, probs_batch = output_batch.cpu(), probs_batch.cpu()
            for text, output, probs in zip(text_batch, output_batch, probs_batch):
                seq_len = get_len_util_stop(output, self.phoneme_tokenizer.end_index)
                predictions[text] = (output[:seq_len], probs[:seq_len])

        return predictions

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device='cpu') -> 'Predictor':
        model, checkpoint = load_checkpoint(checkpoint_path, device=device)
        preprocessor = checkpoint['preprocessor']
        return Predictor(model=model, preprocessor=preprocessor)
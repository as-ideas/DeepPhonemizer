from typing import Dict, Any, List, Tuple, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence

from dp.model import load_checkpoint
from dp.model_utils import get_len_util_stop
from dp.text import Preprocessor
from dp.utils import batchify, get_sequence_prob


class Prediction:

    def __init__(self,
                 word: str,
                 phonemes: str,
                 tokens: List[int],
                 confidence: float,
                 token_probs: List[float]) -> None:
        """
        Container for single word prediction.

        :param word: Original word to predict.
        :param phonemes: Predicted phonemes (without start and end token string).
        :param tokens: Predicted phoneme tokens (including start and end token).
        :param confidence: Total confidence of prediction.
        :param token_probs: Probability of each phoneme token.
        """

        self.word = word
        self.phonemes = phonemes
        self.tokens = tokens
        self.confidence = confidence
        self.token_probs = token_probs


class Predictor:

    def __init__(self,
                 model: torch.nn.Module,
                 preprocessor: Preprocessor) -> None:
        self.model = model
        self.text_tokenizer = preprocessor.text_tokenizer
        self.phoneme_tokenizer = preprocessor.phoneme_tokenizer

    def __call__(self,
                 words: List[str],
                 lang: str,
                 batch_size=8) -> List[Prediction]:
        """
        Predicts phonemes for a list of words.

        :param words: List of words to predict.
        :param lang: Language of texts.
        :param batch_size: Size of batch for model input to speed up inference.
        :return: A list of prediction objects containing (word, phonemes, probs, tokens)
        """

        predictions = dict()
        valid_texts = set()

        # handle words that result in an empty input to the model
        for word in words:
            input = self.text_tokenizer(sentence=word, language=lang)
            decoded = self.text_tokenizer.decode(
                sequence=input, remove_special_tokens=True)
            if len(decoded) == 0:
                predictions[word] = ([], [])
            else:
                valid_texts.add(word)

        valid_texts = sorted(list(valid_texts), key=lambda x: len(x))
        batch_pred = self._predict_batch(texts=valid_texts, batch_size=batch_size,
                                         language=lang)
        predictions.update(batch_pred)

        output = []
        for word in words:
            tokens, probs = predictions[word]
            out_phons = self.phoneme_tokenizer.decode(
                sequence=tokens, remove_special_tokens=True)
            output.append(Prediction(word=word, phonemes=''.join(out_phons),
                                     tokens=tokens, confidence=get_sequence_prob(probs),
                                     token_probs=probs))

        return output

    def _predict_batch(self,
                       texts: List[str],
                       batch_size: int,
                       language: str) \
            -> Dict[str, Tuple[List[int], List[float]]]:
        """ Returns dictionary with key = word and val = Tuple of (phoneme tokens, phoneme probs) """

        predictions = dict()
        text_batches = batchify(texts, batch_size)
        for text_batch in text_batches:
            input_batch, lens_batch = [], []
            for text in text_batch:
                input = self.text_tokenizer(text, language)
                input_batch.append(torch.tensor(input))
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
                predictions[text] = (output[:seq_len].tolist(), probs[:seq_len].tolist())

        return predictions

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device='cpu') -> 'Predictor':
        model, checkpoint = load_checkpoint(checkpoint_path, device=device)
        preprocessor = checkpoint['preprocessor']
        return Predictor(model=model, preprocessor=preprocessor)
import torch
from typing import Dict, Any, List, Tuple, Iterable

from dp.model import TransformerModel
from dp.text import Preprocessor
from dp.utils import load_checkpoint


class Predictor:

    def __init__(self,
                 model: TransformerModel,
                 preprocessor: Preprocessor) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.text_tokenizer = preprocessor.text_tokenizer
        self.phoneme_tokenizer = preprocessor.phoneme_tokenizer

    def __call__(self,
                 texts: List[Iterable[str]],
                 language: str) -> Tuple[List[Iterable[str]], List[Dict[str, Any]]]:
        """
        :param texts: List of texts to predict.
        :param language: Language of texts.
        :return: Predicted phonemes and additional info per prediction such as logits, probability etc.
        """

        predictions = dict()
        valid_texts = set()

        # handle texts that result in an empty input to the model
        for text in texts:
            input = self.text_tokenizer(text, language=language)
            decoded = self.text_tokenizer.decode(input,
                                                 remove_special_tokens=True)
            if len(decoded) == 0:
                predictions[text] = ([], [])
            else:
                valid_texts.add(text)

        # can be batched
        for text in valid_texts:
            input = self.preprocessor.text_tokenizer(text, language)
            input = torch.tensor(input).unsqueeze(0)
            output, logits = self.model.generate(input=input,
                                                 start_index=self.phoneme_tokenizer.get_start_index(language),
                                                 end_index=self.phoneme_tokenizer.end_index)
            predictions[text] = (output, logits[0])

        out_phonemes, out_meta = [], []
        for text in texts:
            output, logits = predictions[text]
            out_phons = self.phoneme_tokenizer.decode(output,
                                                      remove_special_tokens=True)
            out_phonemes.append(out_phons)
            if len(logits) > 0:
                out_meta.append({'phonemes': out_phons, 'logits': logits, 'tokens': output})
            else:
                out_meta.append({'phonemes': out_phons, 'logits': None, 'tokens': output})

        return out_phonemes, out_meta

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device='cpu') -> 'Predictor':
        model, checkpoint = load_checkpoint(checkpoint_path, device=device)
        preprocessor = checkpoint['preprocessor']
        return Predictor(model=model, preprocessor=preprocessor)
import torch
from typing import Dict, Any, List, Tuple, Iterable

from dp.model import TransformerModel
from dp.text import Preprocessor


class Predictor:

    def __init__(self,
                 model: TransformerModel,
                 preprocessor: Preprocessor) -> None:
        self.model = model
        self.preprocessor = preprocessor

    def __call__(self,
                 texts: List[Iterable[str]],
                 language: str) -> Tuple[List[Iterable[str]], List[Dict[str, Any]]]:
        """
        :param texts: List of texts to predict.
        :param language: Language of texts.
        :return: Tuple of predicted phonemes and additional info per prediction such as logits, probability etc.
        """

        text_set = set(texts)
        predictions = dict()
        for text in text_set:
            input = self.preprocessor.text_tokenizer(text)
            input = torch.tensor(input).unsqueeze(0)
            output, logits = self.model.generate(input=input,
                                                 start_index=self.preprocessor.phoneme_tokenizer.start_index,
                                                 end_index=self.preprocessor.phoneme_tokenizer.end_index)
            predictions[text] = (output, logits)

        out_phonemes, out_meta = [], []
        for text in texts:
            output, logits = predictions[text]
            out_phon = self.preprocessor.phoneme_tokenizer.decode(output,
                                                                  remove_special_tokens=True)
            out_phonemes.append(out_phon)
            out_meta.append({'phonemes': out_phon, 'logits': logits, 'tokens': output})

        return out_phonemes, out_meta

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device='cpu') -> 'Predictor':
        device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = TransformerModel.from_config(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model'])
        preprocessor = checkpoint['preprocessor']
        return Predictor(model=model, preprocessor=preprocessor)
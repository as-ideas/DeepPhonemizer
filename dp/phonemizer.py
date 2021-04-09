import torch
from typing import Dict, Union


from dp.model import TransformerModel


# yeah some hard core regex warranted in the future for special chars
class Phonemizer:

    def __init__(self,
                 checkpoint_path: str,
                 lang_phoneme_dict: Dict[str, Dict[str, str]] = None) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model = TransformerModel.from_config(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model'])
        self.preprocessor = checkpoint['preprocessor']
        self.lang_phoneme_dict = lang_phoneme_dict

    def __call__(self, text: str, lang: str) -> str:
        words = text.split()
        output = []
        for word in words:
            phons = self.get_dict_entry(word, lang)
            if phons is None:
                tokens = self.preprocessor.text_tokenizer(word)
                pred = self.model.generate(torch.tensor(tokens).unsqueeze(0))
                pred_decoded = self.preprocessor.phoneme_tokenizer.decode(pred, remove_special_tokens=True)
                phons = ''.join(pred_decoded)
            output.append(phons)
        return ' '.join(output)

    def get_dict_entry(self, word: str, lang: str) -> Union[str, None]:
        if not self.lang_phoneme_dict or lang not in self.lang_phoneme_dict:
            return None
        phoneme_dict = self.lang_phoneme_dict[lang]
        if word in phoneme_dict:
            return phoneme_dict[word]
        elif word.lower() in phoneme_dict:
            return phoneme_dict[word.lower()]
        elif word.title() in phoneme_dict:
            return phoneme_dict[word.title()]
        else:
            return None


if __name__ == '__main__':
    checkpoint_path = '../checkpoints/latest_model.pt'
    phonemizer = Phonemizer(checkpoint_path=checkpoint_path)
    phons = phonemizer('Der kleine Prinzenkönig Francesco Cardinale pillert an seinem Pillermann.', lang='de')
    print(phons)

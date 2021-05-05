from typing import Dict

import torch


class CrossEntropyLoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self,
                pred: torch.tensor,
                batch: Dict[str, torch.tensor]) -> torch.tensor:
        phonemes = batch['phonemes']
        loss = self.criterion(pred.transpose(1, 2), phonemes[:, 1:])
        return loss


class CTCLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion  = torch.nn.CTCLoss()

    def forward(self,
                pred: torch.tensor,
                batch: Dict[str, torch.tensor]) -> torch.tensor:
        pred = pred.transpose(0, 1).log_softmax(2)
        phonemes = batch['phonemes']
        text_len = batch['text_len']
        phon_len = batch['phonemes_len']
        loss = self.criterion(pred, phonemes, text_len, phon_len)
        return loss

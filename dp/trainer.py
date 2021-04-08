from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dp.dataset import new_dataloader
from dp.model import TransformerModel
from dp.text import Tokenizer
from dp.utils import to_device


class Trainer:

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir / 'tensorboard')
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def train(self,
              model: TransformerModel,
              checkpoint: dict,
              train_data: List[tuple],
              val_data: List[tuple]) -> None:

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        config = checkpoint['config']

        optimizer = Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = config['training']['learning_rate']

        train_loader = new_dataloader(data=train_data)
        val_loader = new_dataloader(data=val_data)
        val_batches = sorted([b for b in val_loader], key=lambda x: x['text_len'][0])

        loss_sum = 0.
        start_epoch = model.get_step() // len(train_loader)

        for epoch in range(start_epoch + 1, config['training']['epochs'] + 1):
            pbar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
            for i, batch in pbar:
                batch = to_device(batch, device)
                pbar.set_description(desc=f'Epoch: {epoch} | Step {model.get_step()} '
                                          f'| Loss: {loss_sum / i:#.4}', refresh=True)
                text = batch['text']
                phonemes = batch['phonemes']
                phonemes_in, phonemes_tar = phonemes[:, :-1], phonemes[:, 1:]
                optimizer.zero_grad()
                pred = model.forward(text, phonemes_in)
                loss = self.ce_loss(pred.transpose(1, 2), phonemes_tar)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

                self.writer.add_scalar('Loss/train', loss.item(), global_step=model.get_step())
                self.writer.add_scalar('Params/batch_size', config['training']['batch_size'],
                                       global_step=model.get_step())
                self.writer.add_scalar('Params/learning_rate', config['training']['learning_rate'],
                                       global_step=model.get_step())

                if model.get_step() % config['training']['validate_steps'] == 0:
                    val_loss = self.validate(model, val_batches)
                    self.writer.add_scalar('Loss/val', val_loss, global_step=model.get_step())

                if model.get_step() % config['training']['generate_steps'] == 0:
                    self.generate_samples(model=model,
                                          text_tokenizer=checkpoint['text_tokenizer'],
                                          phoneme_tokenizer=checkpoint['phoneme_tokenizer'],
                                          val_batches=val_batches,
                                          n_samples=config['training']['n_generate_samples'])

                if model.get_step() % config['training']['checkpoint_steps'] == 0:
                    step = model.get_step() // 1000
                    checkpoint['model'] = model.state_dict()
                    checkpoint['optimizer'] = optimizer.state_dict()
                    torch.save(checkpoint, self.checkpoint_dir / f'model_step_{step}k.pt')

            loss_sum = 0
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, self.checkpoint_dir / 'latest_model.pt')

    def validate(self, model: TransformerModel, val_batches: List[dict]) -> float:
        device = next(model.parameters()).device
        model.eval()
        val_loss = 0.
        for batch in val_batches:
            batch = to_device(batch, device)
            text = batch['text']
            phonemes = batch['phonemes']
            phonemes_in, phonemes_tar = phonemes[:, :-1], phonemes[:, 1:]
            with torch.no_grad():
                pred = model.forward(text, phonemes_in)
                loss = self.ce_loss(pred.transpose(1, 2), phonemes_tar)
                val_loss += loss.item()
        model.train()
        return val_loss / len(val_batches)

    def generate_samples(self,
                         model: TransformerModel,
                         text_tokenizer: Tokenizer,
                         phoneme_tokenizer: Tokenizer,
                         val_batches: List[dict],
                         n_samples: int) -> None:
        device = next(model.parameters()).device
        model.eval()
        total_generated = 0
        for batch in val_batches:
            batch = to_device(batch, device)
            for i in range(batch['text'].size(0)):
                total_generated += 1
                text = batch['text'][i, :]
                target = batch['phonemes'][i, :]
                generated = model.generate(text.unsqueeze(0))
                text, target, generated = text.detach().cpu(), target.detach().cpu(), generated.detach().cpu()
                text = text_tokenizer.decode(text, remove_special_tokens=True)
                text = ''.join(text)
                gen_decoded = phoneme_tokenizer.decode(generated, remove_special_tokens=True)
                gen_decoded = ''.join(gen_decoded)
                target = phoneme_tokenizer.decode(target, remove_special_tokens=True)
                target = ''.join(target)
                string = f'     {text} | {target} | {gen_decoded}'
                self.writer.add_text('Text | Target | Prediction', string, global_step=model.get_step())
                if total_generated >= n_samples:
                    model.train()
                    return
        model.train()
        return

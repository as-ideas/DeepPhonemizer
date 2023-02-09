import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from dp.model.model import Model
from dp.model.utils import _trim_util_stop
from dp.preprocessing.text import Preprocessor
from dp.training.dataset import new_dataloader
from dp.training.decorators import ignore_exception
from dp.training.losses import CrossEntropyLoss, CTCLoss
from dp.training.evaluation import evaluate_samples
from dp.utils.io import to_device, unpickle_binary


class Trainer:
    """ Performs model training. """

    def __init__(self, checkpoint_dir: Path, device: torch.device, rank: int, use_ddp: bool, loss_type='ctc') -> None:
        """
        Initializes a Trainer object.

        Args:
          checkpoint_dir (Path): Directory to store the model checkpoints.
          device (torch.device): Device used for training
          rank (int): Rank of the current device
          use_ddp (bool): Flag whether DDP is used for training
          loss_type (str): Type of loss: 'ctc' for forward transformer models
                           and 'cross_entropy' for autoregressive models.
        """

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_ddp = use_ddp
        self.rank = rank
        self.device = device
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'logs'))
        self.loss_type = loss_type
        if loss_type == 'ctc':
            self.criterion = CTCLoss()
        elif loss_type == 'cross_entropy':
            self.criterion = CrossEntropyLoss()
        else:
            raise ValueError(f'Loss not supported: {loss_type}')

    def train(self,
              model: Model,
              checkpoint: Dict[str, Any],
              store_phoneme_dict_in_model: bool = True) -> None:
        """
        Performs training of a transformer model.

        Args:
          model (Model): Model to be trained (can be a fresh model or restored from a checkpoint).
          checkpoint (Dict[str, Any]): Dictionary with entries 'optimizer': optimizer state dict,
                                       'preprocessor': Preprocessor and 'config': Dict.
          store_phoneme_dict_in_model (bool): Whether to store a dictionary of word-phoneme mappings
                                              in the model checkpoint so that it can be automatically
                                              loaded by a Phonemizer object.

        Returns:
          None: the checkpoints will be stored in a folder provided when instantiating a Trainer.
        """

        config = checkpoint['config']
        data_dir = Path(config['paths']['data_dir'])

        model = model.to(self.device)
        model.train()

        if self.use_ddp:
            model = DistributedDataParallel(model, device_ids=[self.rank])

        criterion = self.criterion.to(self.device)

        optimizer = Adam(model.parameters())
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = config['training']['learning_rate']

        train_loader = new_dataloader(dataset_file=data_dir / 'train_dataset.pkl', use_ddp=self.use_ddp,
                                      drop_last=True, batch_size=config['training']['batch_size'])

        if self.rank == 0:
            val_loader = new_dataloader(dataset_file=data_dir / 'val_dataset.pkl', use_ddp=False,
                                        drop_last=False, batch_size=config['training']['batch_size_val'])

            val_batches = sorted([b for b in val_loader], key=lambda x: -x['text_len'][0])

        if store_phoneme_dict_in_model:
            phoneme_dict = unpickle_binary(data_dir / 'phoneme_dict.pkl')
            checkpoint['phoneme_dict'] = phoneme_dict

        scheduler = ReduceLROnPlateau(optimizer,
                                      factor=config['training']['scheduler_plateau_factor'],
                                      patience=config['training']['scheduler_plateau_patience'],
                                      mode='min')
        losses = []
        best_per = math.inf
        if 'step' not in checkpoint:
            checkpoint['step'] = 0
        start_epoch = checkpoint['step'] // len(train_loader)

        for epoch in range(start_epoch + 1, config['training']['epochs'] + 1):
            if self.use_ddp:
                train_loader.sampler.set_epoch(epoch)
            pbar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
            for i, batch in pbar:
                checkpoint['step'] += 1
                step = checkpoint['step']
                self._set_warmup_lr(optimizer=optimizer, step=step,
                                    config=config)
                batch = to_device(batch, self.device)
                avg_loss = sum(losses) / len(losses) if len(losses) > 0 else math.inf
                pbar.set_description(desc=f'Rank: {self.rank} | Epoch: {epoch} | Step {step} '
                                          f'| Loss: {avg_loss:#.4}', refresh=True)

                pred = model(batch)
                loss = criterion(pred, batch)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())

                self.writer.add_scalar('Loss/train', loss.item(), global_step=step)
                self.writer.add_scalar('Params/batch_size', config['training']['batch_size'],
                                       global_step=step)
                self.writer.add_scalar('Params/learning_rate', [g['lr'] for g in optimizer.param_groups][0],
                                       global_step=step)

                if self.rank == 0:
                    if step % config['training']['validate_steps'] == 0:
                        val_loss = self._validate(model, val_batches)
                        self.writer.add_scalar('Loss/val', val_loss, global_step=step)

                    if step % config['training']['generate_steps'] == 0:
                        lang_samples = self._generate_samples(model=model,
                                                              preprocessor=checkpoint['preprocessor'],
                                                              val_batches=val_batches)
                        eval_result = evaluate_samples(lang_samples=lang_samples)
                        self._write_summaries(lang_samples=lang_samples,
                                              eval_result=eval_result,
                                              n_generate_samples=config['training']['n_generate_samples'],
                                              step=step)
                        if eval_result['mean_per'] is not None and eval_result['mean_per'] < best_per:
                            self._save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                                             path=self.checkpoint_dir / f'best_model.pt')
                            self._save_model(model=model, optimizer=None, checkpoint=checkpoint,
                                             path=self.checkpoint_dir / f'best_model_no_optim.pt')
                            scheduler.step(eval_result['mean_per'])

                    if step % config['training']['checkpoint_steps'] == 0:
                        step = step // 1000
                        self._save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                                         path=self.checkpoint_dir / f'model_step_{step}k.pt')

            losses = []
            if self.rank == 0:
                self._save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                                 path=self.checkpoint_dir / 'latest_model.pt')

    def _validate(self, model: Model, val_batches: List[dict]) -> float:
        device = next(model.parameters()).device
        criterion = self.criterion.to(device)
        model.eval()
        val_losses = []
        for batch in val_batches:
            batch = to_device(batch, device)
            with torch.no_grad():
                pred = model(batch)
                loss = criterion(pred, batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_losses.append(loss.item())
        model.train()
        return sum(val_losses) / len(val_losses)

    @ignore_exception
    def _generate_samples(self,
                          model: Model,
                          preprocessor: Preprocessor,
                          val_batches: List[dict]) -> Dict[str, List[Tuple[List[str], List[str], List[str]]]]:

        """ Returns a dictionary with entries lang: Tuple of (word, generated, target) """

        device = next(model.parameters()).device
        model.eval()
        text_tokenizer = preprocessor.text_tokenizer
        phoneme_tokenizer = preprocessor.phoneme_tokenizer
        lang_tokenizer = preprocessor.lang_tokenizer
        lang_prediction_result = dict()

        for batch in val_batches:
            batch = to_device(batch, device)
            generated_batch, _ = (model.module if self.use_ddp else model).generate(batch)
            for i in range(batch['text'].size(0)):
                text_len = batch['text_len'][i]
                text = batch['text'][i, :text_len]
                target = batch['phonemes'][i, :]
                lang = batch['language'][i]
                lang = lang_tokenizer.decode(lang.detach().cpu().item())
                generated = generated_batch[i, :].cpu()
                generated = _trim_util_stop(generated, phoneme_tokenizer.end_index)
                text, target = text.detach().cpu(), target.detach().cpu()
                text = text_tokenizer.decode(text, remove_special_tokens=True)
                generated = phoneme_tokenizer.decode(generated, remove_special_tokens=True)
                target = phoneme_tokenizer.decode(target, remove_special_tokens=True)
                lang_prediction_result[lang] = lang_prediction_result.get(lang, []) + [(text, generated, target)]

        model.train()

        return lang_prediction_result

    @ignore_exception
    def _write_summaries(self,
                         lang_samples: Dict[str, List[Tuple[List[str], List[str], List[str]]]],
                         eval_result: Dict[str, Any],
                         n_generate_samples: int,
                         step: int) -> None:

        self.writer.add_scalar(f'Phoneme_Error_Rate/mean',
                               eval_result['mean_per'], global_step=step)
        self.writer.add_scalar(f'Word_Error_Rate/mean',
                               eval_result['mean_wer'], global_step=step)

        for lang in lang_samples.keys():
            result = eval_result[lang]
            self.writer.add_scalar(f'Phoneme_Error_Rate/{lang}',
                                   result['per'], global_step=step)
            self.writer.add_scalar(f'Word_Error_Rate/{lang}',
                                   result['wer'], global_step=step)

        for lang, samples in lang_samples.items():
            samples = [(''.join(w), ''.join(p), ''.join(t)) for w, p, t in samples]
            word_counts = Counter([word for word, _, _ in samples])
            samples_dedup = [(w, p, t) for w, p, t in samples if word_counts[w] == 1]
            log_texts = dict()
            for word, pred, target in samples_dedup:
                log_texts[word] = f'     {word:<30} {pred:<30} {target:<30}'
            log_text_items = sorted(log_texts.items(), key=lambda x: -len(x[0]))
            log_text_list = [v for k, v in log_text_items]
            log_text = '\n'.join(log_text_list[:n_generate_samples])
            self.writer.add_text(f'{lang}/text_prediction_target', log_text, global_step=step)

    def _save_model(self,
                    model: torch.nn.Module,
                    optimizer: torch.optim,
                    checkpoint: Dict[str, Any],
                    path: Path) -> None:
        checkpoint['model'] = (model.module if self.use_ddp else model).state_dict()
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        else:
            checkpoint['optimizer'] = None
        torch.save(checkpoint, str(path))

    def _set_warmup_lr(self,
                       optimizer: torch.optim,
                       step: int,
                       config: Dict[str, Any]) -> None:

        warmup_steps = config['training']['warmup_steps']
        if warmup_steps > 0 and step <= warmup_steps:
            warmup_factor = 1.0 - max(warmup_steps - step, 0) / warmup_steps
            for g in optimizer.param_groups:
                g['lr'] = config['training']['learning_rate'] * warmup_factor

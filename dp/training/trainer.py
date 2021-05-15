import math
from pathlib import Path
from typing import List, Dict, Any

import torch
import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from dp.model.model import Model
from dp.model.utils import trim_util_stop
from dp.preprocessing.text import Preprocessor
from dp.training.dataset import new_dataloader
from dp.training.decorators import ignore_exception
from dp.training.losses import CrossEntropyLoss, CTCLoss
from dp.training.metrics import phoneme_error, word_error
from dp.utils.io import to_device, unpickle_binary


class Trainer:

    def __init__(self, checkpoint_dir: Path, loss_type='ctc') -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
              checkpoint: dict,
              store_phoneme_dict_in_model=True) -> None:

        config = checkpoint['config']
        data_dir = Path(config['paths']['data_dir'])

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        model.train()

        criterion = self.criterion.to(device)

        optimizer = Adam(model.parameters())
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = config['training']['learning_rate']

        train_loader = new_dataloader(dataset_file=data_dir / 'train_dataset.pkl',
                                      drop_last=True, batch_size=config['training']['batch_size'])
        val_loader = new_dataloader(dataset_file=data_dir / 'val_dataset.pkl',
                                    drop_last=False, batch_size=config['training']['batch_size_val'])
        if store_phoneme_dict_in_model:
            phoneme_dict = unpickle_binary(data_dir / 'phoneme_dict.pkl')
            checkpoint['phoneme_dict'] = phoneme_dict

        val_batches = sorted([b for b in val_loader], key=lambda x: -x['text_len'][0])

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
            pbar = tqdm.tqdm(enumerate(train_loader, 1), total=len(train_loader))
            for i, batch in pbar:
                checkpoint['step'] += 1
                step = checkpoint['step']
                self._set_warmup_lr(optimizer=optimizer, step=step,
                                    config=config)
                batch = to_device(batch, device)
                avg_loss = sum(losses) / len(losses) if len(losses) > 0 else math.inf
                pbar.set_description(desc=f'Epoch: {epoch} | Step {step} '
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

                if step % config['training']['validate_steps'] == 0:
                    val_loss = self.validate(model, val_batches)
                    self.writer.add_scalar('Loss/val', val_loss, global_step=step)

                if step % config['training']['generate_steps'] == 0:
                    per = self.generate_samples(model=model,
                                                preprocessor=checkpoint['preprocessor'],
                                                val_batches=val_batches,
                                                n_log_samples=config['training']['n_generate_samples'],
                                                step=step)
                    if per is not None and per < best_per:
                        self.save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                                        path=self.checkpoint_dir / f'best_model.pt')
                        self.save_model(model=model, optimizer=None, checkpoint=checkpoint,
                                        path=self.checkpoint_dir / f'best_model_no_optim.pt')
                        scheduler.step(per)

                if step % config['training']['checkpoint_steps'] == 0:
                    step = step // 1000
                    self.save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                                    path=self.checkpoint_dir / f'model_step_{step}k.pt')

            losses = []
            self.save_model(model=model, optimizer=optimizer, checkpoint=checkpoint,
                            path=self.checkpoint_dir / 'latest_model.pt')

    def validate(self, model: Model, val_batches: List[dict]) -> float:
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
    def generate_samples(self,
                         model: Model,
                         preprocessor: Preprocessor,
                         val_batches: List[dict],
                         n_log_samples: int,
                         step: int) -> float:
        """ Generates samples and calculates some metrics. Returns phoneme error rate. """

        device = next(model.parameters()).device
        model.eval()
        text_tokenizer = preprocessor.text_tokenizer
        phoneme_tokenizer = preprocessor.phoneme_tokenizer
        lang_tokenizer = preprocessor.lang_tokenizer
        lang_prediction_result = dict()

        for batch in val_batches:
            batch = to_device(batch, device)
            generated_batch, _ = model.generate(batch)
            for i in range(batch['text'].size(0)):
                text_len = batch['text_len'][i]
                text = batch['text'][i, :text_len]
                target = batch['phonemes'][i, :]
                lang = batch['language'][i]
                lang = lang_tokenizer.decode(lang.detach().cpu().item())
                generated = generated_batch[i, :].cpu()
                generated = trim_util_stop(generated, phoneme_tokenizer.end_index)
                text, target = text.detach().cpu(), target.detach().cpu()
                text = text_tokenizer.decode(text, remove_special_tokens=True)
                generated = phoneme_tokenizer.decode(generated, remove_special_tokens=True)
                target = phoneme_tokenizer.decode(target, remove_special_tokens=True)
                lang_prediction_result[lang] = lang_prediction_result.get(lang, []) + [(text, generated, target)]

        # calculate error rates per language
        lang_phon_err, lang_phon_count, lang_word_err = dict(), dict(), dict()
        languages = sorted(lang_prediction_result.keys())
        for lang in languages:
            log_texts = dict()
            for word, generated, target in lang_prediction_result[lang]:
                word = ''.join(word)
                phon_err, phon_count = phoneme_error(generated, target)
                word_err = word_error(generated, target)
                phon_err_dict = lang_phon_err.setdefault(lang, dict())
                phon_count_dict = lang_phon_count.setdefault(lang, dict())
                word_err_dict = lang_word_err.setdefault(lang, dict())
                best_phon_err, best_phon_count = phon_err_dict.get(word, None), phon_count_dict.get(word, None)
                if best_phon_err is None or phon_err / phon_count < best_phon_err / best_phon_count:
                    phon_err_dict[word] = phon_err
                    phon_count_dict[word] = phon_count
                    word_err_dict[word] = word_err
                    gen_decoded, target = ''.join(generated), ''.join(target)
                    log_texts[word] = f'     {word:<30} {gen_decoded:<30} {target:<30}'
            # print predictionso of longest words
            log_text_items = sorted(log_texts.items(), key=lambda x: -len(x[0]))
            log_text_list = [v for k, v in log_text_items][:n_log_samples]
            self.writer.add_text(f'Text_Prediction_Target/{lang}',
                                 '\n'.join(log_text_list), global_step=step)

        phon_errors, phon_counts, word_errors, word_counts = [], [], [], []
        for lang in languages:
            phon_err = sum(lang_phon_err[lang].values())
            phon_errors.append(phon_err)
            phon_count = sum(lang_phon_count[lang].values())
            phon_counts.append(phon_count)
            word_err = sum(lang_word_err[lang].values())
            word_errors.append(word_err)
            word_count = len(lang_word_err[lang])
            word_counts.append(word_count)
            per = phon_err / phon_count
            wer = word_err / word_count
            self.writer.add_scalar(f'Phoneme_Error_Rate/{lang}', per, global_step=step)
            self.writer.add_scalar(f'Word_Error_Rate/{lang}', wer, global_step=step)
        mean_per = sum(phon_errors) / sum(phon_counts)
        mean_wer = sum(word_errors) / sum(word_counts)
        self.writer.add_scalar(f'Phoneme_Error_Rate/mean', mean_per, global_step=step)
        self.writer.add_scalar(f'Word_Error_Rate/mean', mean_wer, global_step=step)

        model.train()

        return sum(phon_errors) / sum(phon_counts)

    def save_model(self,
                   model: torch.nn.Module,
                   optimizer: torch.optim,
                   checkpoint: Dict[str, Any],
                   path: Path) -> None:
        checkpoint['model'] = model.state_dict()
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

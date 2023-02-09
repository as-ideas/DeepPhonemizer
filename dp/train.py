import os
from pathlib import Path

import torch
from torch.distributed import init_process_group

from dp.model.model import load_checkpoint, ModelType, \
    create_model
from dp.preprocessing.text import Preprocessor
from dp.training.trainer import Trainer
from dp.utils.io import read_config
from dp.utils.logging import get_logger

logger = get_logger(__name__)


def train(rank: int,
          num_gpus: int,
          config_file: str,
          checkpoint_file: str = None) -> None:
    """
    Runs training of a transformer model.

    Args:
      rank (int): Device id
      num_gpus (int): Number of devices
      config_file (str): Path to the config.yaml that stores all necessary parameters.
      checkpoint_file (str, optional): Path to a model checkpoint to resume training for (e.g. latest_model.pt)

    Returns:
        None: The model checkpoints are stored in a folder provided by the config.

    """

    config = read_config(config_file)

    if num_gpus > 1:
        os.environ["MASTER_ADDR"] = config['training']['ddp_host']
        os.environ["MASTER_PORT"] = config['training']['ddp_post']
        init_process_group(backend=config['training']['ddp_backend'], rank=rank, world_size=num_gpus)

    if checkpoint_file is not None:
        logger.info(f'Restoring model from checkpoint: {checkpoint_file}')
        model, checkpoint = load_checkpoint(checkpoint_file)
        model.train()
        step = checkpoint['step']
        logger.info(f'Loaded model with step: {step}')
        for key, val in config['training'].items():
            val_orig = checkpoint['config']['training'][key]
            if val_orig != val:
                logger.info(f'Overwriting training param: {key} {val_orig} --> {val}')
                checkpoint['config']['training'][key] = val
        config = checkpoint['config']
        model_type = config['model']['type']
        model_type = ModelType(model_type)
    else:
        logger.info('Initializing new model from config...')
        preprocessor = Preprocessor.from_config(config)
        model_type = config['model']['type']
        model_type = ModelType(model_type)
        model = create_model(model_type, config=config)
        checkpoint = {
            'preprocessor': preprocessor,
            'config': config,
        }

    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    logger.info(f'Checkpoints will be stored at {checkpoint_dir.absolute()}')
    loss_type = 'cross_entropy' if model_type.is_autoregressive() else 'ctc'

    if num_gpus > 0:
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    use_ddp = True if num_gpus > 1 else False

    trainer = Trainer(checkpoint_dir=checkpoint_dir, device=device, rank=rank, use_ddp=use_ddp, loss_type=loss_type)
    trainer.train(model=model,
                  checkpoint=checkpoint,
                  store_phoneme_dict_in_model=config['training']['store_phoneme_dict_in_model'])

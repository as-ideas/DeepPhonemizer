from pathlib import Path

from dp.model.model import load_checkpoint, ModelType, \
    create_model
from dp.preprocessing.text import Preprocessor
from dp.training.trainer import Trainer
from dp.utils.io import read_config
from dp.utils.logging import get_logger

logger = get_logger(__name__)


def train(config_file: str,
          checkpoint_file: str = None) -> None:

    config = read_config(config_file)
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
    trainer = Trainer(checkpoint_dir=checkpoint_dir, loss_type=loss_type)
    trainer.train(model=model,
                  checkpoint=checkpoint,
                  store_phoneme_dict_in_model=config['training']['store_phoneme_dict_in_model'])
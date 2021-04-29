from dp.model.model import LstmModel, ForwardTransformer, AutoregressiveTransformer, load_checkpoint
from dp.preprocessing.text import Preprocessor
from dp.training.trainer import Trainer
from dp.utils import read_config


def train(config_file: str,
          checkpoint_file: str = None) -> None:

    config = read_config(config_file)
    if checkpoint_file is not None:
        print(f'Restoring model from checkpoint: {checkpoint_file}')
        model, checkpoint = load_checkpoint(checkpoint_file)
        model.train()
        step = checkpoint['step']
        print(f'Loaded model with step: {step}')
        for key, val in config['training'].items():
            val_orig = checkpoint['config']['training'][key]
            if val_orig != val:
                print(f'Overwriting training param: {key} {val_orig} --> {val}')
                checkpoint['config']['training'][key] = val
        config = checkpoint['config']
        model_type = config['model']['type']
    else:
        print('Initializing new model from config...')
        preprocessor = Preprocessor.from_config(config)
        model_type = config['model']['type']
        supported_types = ['lstm', 'transformer', 'autoreg_transformer']

        if model_type == 'lstm':
            model = LstmModel.from_config(config)
        elif model_type == 'transformer':
            model = ForwardTransformer.from_config(config)
        elif model_type == 'autoreg_transformer':
            model = AutoregressiveTransformer.from_config(config)
        else:
            raise ValueError(f'Model type not supported: {model_type}. Supported types: {supported_types}')
        checkpoint = {
            'preprocessor': preprocessor,
            'config': config,
        }

    loss_type = 'cross_entropy' if 'autoreg_' in model_type else 'ctc'
    trainer = Trainer(checkpoint_dir=config['paths']['checkpoint_dir'], loss_type=loss_type)
    trainer.train(model=model,
                  checkpoint=checkpoint,
                  store_phoneme_dict_in_model=config['training']['store_phoneme_dict_in_model'])
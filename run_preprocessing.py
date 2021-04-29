from dp.preprocess import preprocess
from dp.utils import unpickle_binary

if __name__ == '__main__':

    config_file = 'dp/configs/autoreg_config.yaml'
    train_data = unpickle_binary('/Users/cschaefe/datasets/nlp/de_us_phonemes_train.pkl')
    val_data = unpickle_binary('/Users/cschaefe/datasets/nlp/de_us_phonemes_val.pkl')

    preprocess(config_file=config_file, train_data=train_data)


from dp.preprocess import preprocess
from dp.train import train

if __name__ == '__main__':

    train_data = [
                     ('en_us', 'young', 'jʌŋ'),
                     ('de', 'benützten', 'bənʏt͡stn̩'),
                     ('de', 'gewürz', 'ɡəvʏʁt͡s')
                 ] * 1000

    config_file = 'dp/configs/forward_config.yaml'
    preprocess(config_file=config_file,
               train_data=train_data,
               deduplicate_train_data=False)
    train(config_file=config_file)
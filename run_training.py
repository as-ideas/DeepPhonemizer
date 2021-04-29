from dp.train import train

if __name__ == '__main__':

    config_file = 'dp/configs/autoreg_config.yaml'
    train(config_file=config_file, checkpoint_file=None)


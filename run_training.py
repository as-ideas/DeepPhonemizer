from dp.train import train

if __name__ == '__main__':

    config_file = 'config_multilang.yaml'
    train(config_file=config_file, checkpoint_file='checkpoints/multilang/best_model.pt')


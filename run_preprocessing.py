from dp.preprocess import preprocess
from dp.utils.io import unpickle_binary, read_config

if __name__ == '__main__':

    config_file = 'config.yaml'
    with open ('/Users/cschaefe/datasets/nlp/cmudict-0.7b-ipa.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    config = read_config(config_file)
    text_symb = set(config['preprocessing']['text_symbols'])

    train_data = []
    word_set = set()
    for line in lines:
        word = line.split('\t')[0].lower()
        word = ''.join([c for c in word if c in text_symb])
        if word in word_set:
            continue
        phons = line.split('\t')[1].split(',')
        for p in phons:
            p = p.replace('\n', '')
            train_data.append(('en_us', word, p))
        word_set.add(word)
    preprocess(config_file=config_file, train_data=train_data, val_data=None)


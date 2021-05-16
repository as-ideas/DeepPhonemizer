from dp.preprocess import preprocess
from dp.utils.io import unpickle_binary, read_config

def get_data(file, lang):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        split = line.split('\t')
        word = split[0]
        phon = split[1].replace(' ', '').replace('\n', '')
        data.append((lang, word, phon))
    return data


if __name__ == '__main__':

    config_file = 'config_multilang.yaml'

    files = [
        ('en_uk', '/Users/cschaefe/workspace/wikipron/data/scrape/tsv/eng_latn_uk_phonemic_filtered.tsv'),
        ('en_us', '/Users/cschaefe/workspace/wikipron/data/scrape/tsv/eng_latn_us_phonemic_filtered.tsv'),
        ('de', '/Users/cschaefe/workspace/wikipron/data/scrape/tsv/ger_latn_phonemic_filtered.tsv'),
        ('fr', '/Users/cschaefe/workspace/wikipron/data/scrape/tsv/fre_latn_phonemic_filtered.tsv'),
        ('es', '/Users/cschaefe/workspace/wikipron/data/scrape/tsv/spa_latn_ca_phonemic_filtered.tsv')
    ]

    all_data = []
    for lang, file in files:
        data = get_data(file, lang)
        all_data.extend(data)

    text_symb, phon_symb = set(), set()
    for l, w, p in all_data:
        text_symb.update(list(w))
        phon_symb.update(list(p))

    text_symb = sorted(list(text_symb))
    phon_symb = sorted(list(phon_symb))

    print(f'text_symbols: {text_symb}')
    print(f'phoneme_symbols: {phon_symb}')

    preprocess(config_file=config_file, train_data=all_data, val_data=None)


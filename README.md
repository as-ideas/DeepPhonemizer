# DeepPhonemizer

DeepPhonemizer is a library for grapheme to phoneme conversion based on Transformer models. 
It is intended to be used in text-to-speech production systems due to its accuracy and efficiency.
The main advantages of this repo are:

* Easy-to-use API for inference and training.
* Multilingual: You can train a single model on several languages.
* Speed: Using the forward Transformer model, phonemization of large articles on a CPU is almost instantaneous.

You can choose between a forward Transformer model (trained with CTC) and its autoregressive
counterpart. The former is faster and more stable whether the latter is slightly more accurate
in terms of smaller word error rate. 

Check out the training and inference tutorial on [colab](https://colab.research.google.com/github/as-ideas/DeepPhonemizer/blob/master/notebooks/Training_Example.ipynb)!


## ⚙️ Installation

```bash
pip install deep-phonemizer
```

### Training

All parameters are set in a config.yaml, you can find a config in the installed package under:
```bash
dp/configs/forward_config.yaml
```

Pepare data in a tuple-format and use the preprocess and train API:

```python
from dp.preprocess import preprocess
from dp.train import train

train_data = [
                ('en_us', 'young', 'jʌŋ'),
                ('de', 'benützten', 'bənʏt͡stn̩'),
                ('de', 'gewürz', 'ɡəvʏʁt͡s')
             ] * 100


preprocess(config_file='config.yaml', train_data=train_data)
train(config_file='config.yaml')
```
Model checkpoints will be stored in the checkpoints path that is provided by the config.yaml.

### Inference

Load a phonemizer from a checkpoint and run a prediction. By default, the phonemizer stores a 
dictionary of word-phoneme mappings that is applied first, and it uses the Transformer model
only to predict out-of-dictionary words.

```python
from dp.phonemizer import Phonemizer

phonemizer = Phonemizer.from_checkpoint('/content/checkpoints/best_model.pt')
result = phonemizer('Phonemizing an English text is imposimpable!', lang='en_us')
print(result)
```

If you need more inference information, you can use following API:

```python
from dp.phonemizer import Phonemizer
result = phonemizer.phonemise_list(['Phonemizing an US-English text is imposimpable!'], lang='en_us')

for word, pred in result.predictions.items():
  print(f'{word} {pred.phonemes} {pred.confidence}')
```


## 🏗 Maintainers
* Christian Schäfer, github: [cschaefer26](https://github.com/cschaefer26)


## References

[Transformer based Grapheme-to-Phoneme Conversion](https://arxiv.org/abs/2004.06338)
[GRAPHEME-TO-PHONEME CONVERSION USING
LONG SHORT-TERM MEMORY RECURRENT NEURAL NETWORKS](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43264.pdf)

import numpy
from typing import List, Union


def phoneme_error_rate(text: List[Union[str, int]], phonemes: List[Union[str, int]]) -> float:
    d = numpy.zeros((len(phonemes) + 1) * (len(text) + 1),
                    dtype=numpy.uint8)
    d = d.reshape((len(phonemes) + 1, len(text) + 1))
    for i in range(len(phonemes) + 1):
        for j in range(len(text) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(phonemes) + 1):
        for j in range(1, len(text) + 1):
            if phonemes[i - 1] == text[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(phonemes)][len(text)] / float(len(phonemes))


if __name__ == '__main__':

    pred = list('abdc')
    gold = list('abc')
    wer = phoneme_error_rate(pred, gold)

    print(wer)
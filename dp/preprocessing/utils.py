import math
from typing import List, Union, Any


def product(probs: Union[None, List[float]]) -> float:
    """
    Calculates the product of a list of probabilities.
    :param probs: Probabilities to be multiplied.
    :return: Product of probabilities.
    """

    if probs is None or len(probs) == 0:
        return 0.
    if 0 in probs:
        return 0
    prob = math.exp(sum([math.log(p) for p in probs]))
    return prob


def batchify(input: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Generates batches out of a list of inputs.

    :param input: List of input values.
    :param batch_size: Size of batches to generate.
    :return: List of batches of size batch_size (the last batch may be shorter).
    """

    l = len(input)
    output = []
    for i in range(0, l, batch_size):
        batch = input[i:min(i + batch_size, l)]
        output.append(batch)
    return output


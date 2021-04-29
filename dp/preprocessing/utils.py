import math
from typing import List

import torch


def get_sequence_prob(probs: torch.tensor) -> float:
    if probs is None or len(probs) == 0:
        return 0.
    if 0 in probs:
        return 0
    prob = math.exp(sum([math.log(p) for p in probs]))
    return prob


def batchify(input: list, batch_size: int) -> List[list]:
    l = len(input)
    output = []
    for i in range(0, l, batch_size):
        batch = input[i:min(i + batch_size, l)]
        output.append(batch)
    return output


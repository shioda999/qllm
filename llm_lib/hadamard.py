import torch
import numpy as np
from scipy.linalg import hadamard
import math

def _generate_hadamard_matrix(n, device):
    # nは2のべき乗である必要があります
    if not (n > 0 and ((n & (n - 1)) == 0)):
        raise ValueError("n must be a power of 2")
    H = hadamard(n) / math.sqrt(n)
    return torch.tensor(H, device=device)

def generate_hadamard_matrix(n, device):
    k = 1
    m = []
    while n > 0:
        if (n & 1):
            m.append(_generate_hadamard_matrix(k, device))
        n >>= 1
        k <<= 1
    m.reverse()
    r = torch.block_diag(*m)
    return r
import math

import torch
import scipy.integrate as integrate

from base import Transform, CompressionPipeline, flatten
from hadamard import RandomizedHadamard, padToPowerOf2
from quantization_constants import QuantizationType, get_all_quantization_constants_tensors
from random_p import RandomP, TopP


ONE_BIT_SIGMA_LLOYD = {
    0: 0.79788,
    1: 1.52514,
    2: 2.37322,
    3: 3.2831,
    4: 4.22561,
    5: 5.1865,
    6: 6.15848,
    7: 7.13755,
    8: 8.12137,
    9: 9.10852,
    10: 10.0981,
}


def solve_lloyd_(left, n, steps=1000):
    if n == 1 and isinstance(left, int):
        return [0, ONE_BIT_SIGMA_LLOYD[left]], [0, left, float("inf")]
    boundaries = [left] + [left + i for i in range(1,n)] + [float("inf")]
    for i in range(steps):
        centers_of_mass = [integrate.quad(lambda x: x * math.exp(-x**2 / 2 + a**2/2), a, b)[0] / integrate.quad(lambda x: math.exp(-x**2 / 2 + a**2/2), a, b)[0] for a,b in zip(boundaries[:-1], boundaries[1:])]
        boundaries = [left] + [(a + b) / 2 for a, b in zip(centers_of_mass[:-1], centers_of_mass[1:])] + [float("inf")]

    return [0] + centers_of_mass, [0] + boundaries
        

class SigmaQuantization(Transform):
    def __init__(self, sigmas: float, bits: int, device="cpu"):
        self.sigmas = sigmas
        self.bits = bits

        if self.bits == 1 and not isinstance(self.sigmas, int):
            self.center_of_mass = integrate.quad(lambda x: x * math.exp(-x**2 / 2), sigmas, math.inf)[0] / integrate.quad(lambda x: math.exp(-x**2 / 2), sigmas, math.inf)[0]
        else:
            centers_of_mass, boundaries = solve_lloyd_(sigmas, 2**bits//2)
            self.centers_of_mass = torch.tensor(centers_of_mass, device=device)
            self.boundaries = torch.tensor(boundaries, device=device)


    def forward(self, x):
        result = torch.zeros_like(x)
        if self.bits == 1 and not isinstance(self.sigmas, int):
            result[x > self.sigmas] = self.center_of_mass
            result[x < -self.sigmas] = -self.center_of_mass
        else:
            ids = torch.bucketize(x.abs(), self.boundaries, right=False) - 1
            quantized_absolute_values = torch.take(self.centers_of_mass, ids)
            result = quantized_absolute_values * torch.sign(x)

        return result, None

    def backward(self, x, context):
        return x, torch.count_nonzero(x).item()


def bernoulli_mask(shape, device, p, prng):
    return torch.empty(shape, dtype=torch.bool, device=device).bernoulli_(p=p, generator=prng)


def mask_split(x, mask):
    x0 = torch.masked_select(x, torch.logical_not(mask))
    x1 = torch.masked_select(x, mask)
    return x0, x1


def mask_combine(x0, x1, mask):
    x = torch.empty(mask.shape, dtype=x0.dtype, device=x0.device)
    x.masked_scatter_(torch.logical_not(mask), x0)
    x.masked_scatter_(mask, x1)

    return x


def sum_squares(x):
    return torch.sum(x**2)


def l2(x):
    return torch.sqrt(sum_squares(x))


def rot_sigma_builder(sigmas: float, bits: int, device="cpu"):
    """

    Args:
      bits: A positive real bit rate
      q_type: Either 'max_lloyd' (Section 3) or 'ee' (Section 4.3)
      device: Torch device to use

    Returns:
      EDEN compression scheme instance
    """
    transforms = [flatten]
    transforms += [
        padToPowerOf2,
        RandomizedHadamard(device),
        SigmaQuantization(sigmas, bits, device),
    ]
    return CompressionPipeline(transforms)

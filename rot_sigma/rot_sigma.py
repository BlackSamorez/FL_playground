import math

import torch
import scipy.integrate as integrate

from base import Transform, CompressionPipeline, flatten
from hadamard import RandomizedHadamard, padToPowerOf2
from quantization_constants import QuantizationType, get_all_quantization_constants_tensors


def solve_lloyd_(left, n, steps=1000):
    boundaries = [left] + [left + i for i in range(1,n)] + [float("inf")]
    for i in range(steps):
        centers_of_mass = [integrate.quad(lambda x: x * math.exp(-x**2 / 2 + a**2/2), a, b)[0] / integrate.quad(lambda x: math.exp(-x**2 / 2 + a**2/2), a, b)[0] for a,b in zip(boundaries[:-1], boundaries[1:])]
        boundaries = [left] + [(a + b) / 2 for a, b in zip(centers_of_mass[:-1], centers_of_mass[1:])] + [float("inf")]

    return [0] + centers_of_mass, boundaries
        

class SigmaQuantization(Transform):
    def __init__(self, sigmas: float, bits: int, device="cpu"):
        self.sigmas = sigmas
        self.bits = bits

        centers_of_mass, boundaries = solve_lloyd_(sigmas, 2**bits//2)
        self.centers_of_mass = torch.tensor(centers_of_mass, device=device)
        self.boundaries = torch.tensor(boundaries, device=device)


    def forward(self, x):
        d = x.numel()
        scale = math.sqrt(d) / l2(x)
        normalized_x = x * scale
        ids = torch.bucketize(normalized_x.abs(), self.boundaries, right=False)
        quantized_absolute_values = torch.take(self.centers_of_mass, ids)
        result = quantized_absolute_values * torch.sign(x)
        result /= scale
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


class BiasedRandomP(Transform):
  """
  Random Sparsification given a parameter p remaining to determine the
  probability of a coordinate to keep
  """

  def __init__(self, p=0.5, device='cpu'):
    self.prng = torch.Generator(device=device)
    self.p = p

  def forward(self, x):
    seed = self.prng.seed()
    original_shape = x.shape
    original_d = x.numel()
    mask = torch.empty_like(x).bernoulli_(p=self.p, generator=self.prng)

    indices = torch.nonzero(mask, as_tuple=True)
    return x[indices], (seed, original_shape, original_d)

  def backward(self, sparse_x, context):
    seed, original_shape, original_d = context

    x = torch.zeros(original_d, dtype=sparse_x.dtype, layout=sparse_x.layout,
                    device=sparse_x.device)

    indices = torch.nonzero(
      torch.empty_like(x).bernoulli_(p=self.p, generator=self.prng.manual_seed(seed))
    ).squeeze()
    x.scatter_(0, indices, sparse_x)

    return x.view(original_shape), len(indices)


def rot_sigma_builder(p:float, sigmas: float, bits: int, device="cpu"):
    """

    Args:
      bits: A positive real bit rate
      q_type: Either 'max_lloyd' (Section 3) or 'ee' (Section 4.3)
      device: Torch device to use

    Returns:
      EDEN compression scheme instance
    """
    transforms = [
        flatten,
        BiasedRandomP(p,device),
        padToPowerOf2,
        RandomizedHadamard(device),
        SigmaQuantization(sigmas, bits, device),
    ]
    return CompressionPipeline(transforms)

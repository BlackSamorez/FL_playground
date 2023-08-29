import math
from abc import abstractmethod
from functools import lru_cache
from typing import Literal

import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn.functional as F
from scipy.linalg import inv
from scipy.special import erf, erfc
from scipy.stats import ortho_group
from torch import FloatTensor

from MiniFL.message import Message

from .basic import RandKBaseCompressor
from .interfaces import Compressor, ContractiveCompressor, UnbiasedCompressor

QuantizationType = Literal["max_lloyd", "ee"]


class TopSigmaCompressor(ContractiveCompressor):
    def __init__(self, size: int, sigmas: float, bits: int = 1, real_rotation=False, device="cpu", seed=0):
        super().__init__(size=size)
        self.sigmas = sigmas
        self.bits = bits

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Rotation
        self.real_rotation = real_rotation

        # Quantization
        centers_of_mass, boundaries = solve_lloyd_(sigmas, 2**bits // 2)
        self.centers_of_mass = torch.tensor(centers_of_mass, device=device)
        self.boundaries = torch.tensor(boundaries, device=device)

    def compress(self, x: FloatTensor) -> Message:
        compression_result = self.inner_compress(x)
        return Message(
            data=self.inner_decompress(compression_result),
            size=self.count_bits(compression_result),
        )

    def alpha(self) -> float:
        return (
            1
            - np.sqrt(2 / np.pi)
            * (np.sqrt(np.pi / 2) * erf(self.sigmas / np.sqrt(2)) - np.exp(-self.sigmas**2 / 2) * self.sigmas)
            - (
                -2 * np.exp(-self.sigmas**2) / (np.pi * erfc(self.sigmas / np.sqrt(2)))
                + erfc(self.sigmas / np.sqrt(2))
                + np.sqrt(2 / np.pi) * np.exp(-self.sigmas**2 / 2) * self.sigmas
            )
        )

    def count_bits(self, compression_result) -> float:
        nonzero = (compression_result["assignments"] != 0).sum().item()
        size = compression_result["quantized_size"]

        return (
            nonzero * self.bits
            + 2  # top type
            + 32  # scale
            + min(
                nonzero * np.log2(size),  # positions
                (size - nonzero) * np.log2(size),  # not positions
                size,  # mask
            )
        )

    def inner_compress(self, x: FloatTensor):
        compression_result = {}

        # Flatten
        original_shape = x.shape
        compression_result["original_shape"] = original_shape
        data = x.flatten()

        # Rotate
        if self.real_rotation:
            pre_rotation_size = data.shape[0]
            compression_result["pre_rotation_size"] = pre_rotation_size
            rotation_seed = self.generator.seed() % 2**32  # TODO: get_state()
            compression_result["rotation_seed"] = rotation_seed
            np.random.seed(seed=rotation_seed)
            data = torch.from_numpy(ortho_group.rvs(pre_rotation_size) @ data.numpy()).to(data.device).to(data.dtype)
        else:
            unpadded_size = data.numel()
            compression_result["unpadded_size"] = unpadded_size
            if unpadded_size & (unpadded_size - 1) != 0:
                dim_with_pad = 2 ** (math.floor(math.log2(unpadded_size)) + 1)
                data = F.pad(data, (0, dim_with_pad - unpadded_size))

            rotation_seed = self.generator.get_state()
            compression_result["rotation_seed"] = rotation_seed
            data = randomized_hadamard_transform_(data, self.generator)

        # Quantize
        quantized_size = data.numel()
        compression_result["quantized_size"] = quantized_size
        scale = math.sqrt(quantized_size) / l2(data)
        compression_result["scale"] = scale
        normalized_data = data * scale
        compression_result["assignments"] = torch.bucketize(normalized_data.abs(), self.boundaries, right=False)
        compression_result["signs"] = torch.sign(data)

        return compression_result

    def inner_decompress(self, compression_result) -> FloatTensor:
        # Dequantize
        data = torch.take(self.centers_of_mass, compression_result["assignments"])
        data = data * compression_result["signs"]
        data /= compression_result["scale"]

        # Rotate back
        if self.real_rotation:
            rotation_seed = compression_result["rotation_seed"]
            pre_rotation_size = compression_result["pre_rotation_size"]
            np.random.seed(seed=rotation_seed)
            data = torch.from_numpy(inv(ortho_group.rvs(pre_rotation_size)) @ data.numpy()).to(data.device)
        else:
            rotation_seed = compression_result["rotation_seed"]
            data = inverse_randomized_hadamard_transform_(data, self.generator.set_state(rotation_seed))

            unpadded_size = compression_result["unpadded_size"]
            data = data[:unpadded_size]

        # Unflatten
        original_shape = compression_result["original_shape"]
        x = data.view(original_shape)

        return x


### Hadamard


def hadamard_transform_(vec):
    """fast Walsh–Hadamard transform (in-place)

    :param vec: vec is expected to be a power of 2!
    :return: the Hadamard transform of vec
    """
    d = vec.numel()
    original_shape = vec.shape
    h = 2
    while h <= d:
        hf = h // 2
        vec = vec.view(d // h, h)

        ## the following is a more inplace way of doing the following:
        # half_1 = batch[:, :, :hf]
        # half_2 = batch[:, :, hf:]
        # batch = torch.cat((half_1 + half_2, half_1 - half_2), dim=-1)
        # the NOT inplace seems to be actually be slightly faster
        # (I assume for making more memory-contiguous operations. That being said,
        # it more easily throws out-of-memory and may slow things overall,
        # so using inplace version below:)

        vec[:, :hf] = vec[:, :hf] + vec[:, hf : 2 * hf]
        vec[:, hf : 2 * hf] = vec[:, :hf] - 2 * vec[:, hf : 2 * hf]
        h *= 2

    vec *= d**-0.5  # vec /= np.sqrt(d)

    return vec.view(*original_shape)


def rademacher_like(x, generator):
    """(previously random_diagonal)"""
    return 2 * torch.torch.empty_like(x).bernoulli_(generator=generator) - 1


def randomized_hadamard_transform_(x, generator):
    d = rademacher_like(x, generator)

    return hadamard_transform_(x * d)


def inverse_randomized_hadamard_transform_(tx, generator):
    d = rademacher_like(tx, generator)

    return hadamard_transform_(tx) * d


### Quantization


def solve_lloyd_(left, n, steps=1000):
    boundaries = [left] + [left + i for i in range(1, n)] + [float("inf")]
    for i in range(steps):
        centers_of_mass = [
            integrate.quad(lambda x: x * math.exp(-(x**2) / 2 + a**2 / 2), a, b)[0]
            / integrate.quad(lambda x: math.exp(-(x**2) / 2 + a**2 / 2), a, b)[0]
            for a, b in zip(boundaries[:-1], boundaries[1:])
        ]
        boundaries = [left] + [(a + b) / 2 for a, b in zip(centers_of_mass[:-1], centers_of_mass[1:])] + [float("inf")]

    return [0] + centers_of_mass, boundaries


### Aux
def bernoulli_mask(shape, device, p, generator):
    return torch.empty(shape, dtype=torch.bool, device=device).bernoulli_(p=p, generator=generator)


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

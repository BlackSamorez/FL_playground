import math
from abc import abstractmethod
from functools import cache, lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import inv
from scipy.special import erfinv
from scipy.stats import ortho_group
from torch import FloatTensor

from MiniFL.message import Message

from .interfaces import Compressor, ContractiveCompressor, UnbiasedCompressor


def get_thresholds_(n):
    return torch.tensor(np.linspace(-1, 1, num=n + 2)[1:-1])
    # return torch.tensor(erfinv(2 * np.linspace(0, 1, num=n + 1, endpoint=False)[1:] - 1) * np.sqrt(2))


class EdenCorrelatedCompressor(Compressor):
    def __init__(self, size: int, rank: int, world_size: int, real_rotation=False, device="cpu", seed=0):
        super().__init__(size=size)
        self.rank = rank
        self.world_size = world_size

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Rotation
        self.real_rotation = real_rotation

        # Quantization
        self.thresholds = get_thresholds_(self.world_size)
        self.center_of_mass = 0.7978845608028654

    def compress(self, x: FloatTensor) -> Message:
        compression_result = self.inner_compress(x)
        if compression_result["is_zero"]:
            bits = 32
        else:
            bits = self.size + 32
        return Message(
            data=self.inner_decompress(compression_result),
            size=bits,
        )

    def inner_compress(self, x: FloatTensor):
        compression_result = {}
        if torch.count_nonzero(x) == 0:
            compression_result["is_zero"] = True
            compression_result["original_shape"] = x.shape
            return compression_result
        else:
            compression_result["is_zero"] = False

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
        d = data.numel()
        permutations = torch.empty(dtype=torch.int64, size=(d, self.world_size))
        for i in range(d):
            permutations[i] = torch.randperm(self.world_size, generator=self.generator)
        selected_thresholds = self.thresholds[permutations[:, self.rank]]
        normalized_data = data * math.sqrt(d) / l2(data)
        quantized_data = torch.where(normalized_data > selected_thresholds, self.center_of_mass, -self.center_of_mass)

        scale = sum_squares(data) / (quantized_data @ data)  # l2(data) / math.sqrt(d)
        compression_result["scale"] = scale
        compression_result["quantized_data"] = quantized_data
        return compression_result

    def inner_decompress(self, compression_result) -> FloatTensor:
        if compression_result["is_zero"]:
            return torch.zeros(compression_result["original_shape"], dtype=torch.float32)

        # Dequantize
        data = compression_result["quantized_data"] * compression_result["scale"]

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


def sum_squares(x):
    return torch.sum(x**2)


def l2(x):
    return torch.sqrt(sum_squares(x))

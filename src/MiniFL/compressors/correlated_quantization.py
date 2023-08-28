import torch
from torch import FloatTensor

from MiniFL.message import Message

from .interfaces import InputVarianceCompressor


class CorrelatedQuantizer(InputVarianceCompressor):
    def __init__(self, size: int, rank: int, world_size: int, num_levels: int, seed: int = 0):
        super().__init__(size)
        self.rank = rank
        self.world_size = world_size

        if num_levels != 1:
            raise NotImplementedError("num_levels != 1")
        self.num_levels = num_levels

        self.perm_generator = torch.Generator()
        self.perm_generator.manual_seed(seed)

        self.offset_generator = torch.Generator()
        self.offset_generator.manual_seed(seed + rank)

    def compress(self, x: FloatTensor) -> Message:
        d = x.numel()
        r = torch.linalg.vector_norm(x) ** 2

        x_normalized = (x + r) / (2 * r)

        gammas = torch.rand(d, generator=self.offset_generator) / self.world_size
        permutations = torch.empty(dtype=torch.int64, size=(d, self.world_size))
        for i in range(d):
            permutations[i] = torch.randperm(self.world_size, generator=self.perm_generator)
        permutations = permutations.to(torch.float32)

        x_compressed = torch.zeros_like(x_normalized)
        x_compressed[x_normalized > permutations[:, self.rank] / self.world_size + gammas] = 1

        x_decompressed = 2 * r * x_compressed - r

        return Message(x_decompressed, 32 + d)

    def ab(self) -> (float, float):
        return 48 / self.world_size**2, 0

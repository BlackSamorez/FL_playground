import torch
from torch import FloatTensor

from MiniFL.message import Message

from .interfaces import InputVarianceCompressor


class CorrelatedQuantizer(InputVarianceCompressor):
    def __init__(self, size: int, rank: int, world_size: int, r: float, num_levels: int, update_r: True, seed: int = 0):
        super().__init__(size)
        self.rank = rank
        self.world_size = world_size
        self.r = r

        if num_levels != 1:
            raise NotImplementedError("num_levels != 1")
        self.num_levels = num_levels
        self.update_r = update_r

        self.perm_generator = torch.Generator()
        self.perm_generator.manual_seed(seed)

        self.offset_generator = torch.Generator()
        self.offset_generator.manual_seed(seed + rank)

    def compress(self, x: FloatTensor) -> Message:
        d = x.numel()

        x_normalized = (x + self.r) / (2 * self.r)

        gammas = torch.rand(d, generator=self.offset_generator)
        permutations = torch.empty(dtype=torch.int64, size=(d, self.world_size))
        for i in range(d):
            permutations[i] = torch.randperm(self.world_size, generator=self.perm_generator)
        permutations = permutations.to(torch.float32)

        compressed_x = torch.zeros_like(x_normalized)
        compressed_x[x_normalized > permutations[:, self.rank] / self.world_size + gammas] = 1

        decompressed_x = 2 * self.r * compressed_x - self.r

        return Message(decompressed_x, 32 + d)

    def ab(self) -> (float, float):
        return 48 * self.r**2 / self.world_size**2, 0

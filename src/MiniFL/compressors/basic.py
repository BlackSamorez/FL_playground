import math

import torch
from torch import FloatTensor, Tensor

from MiniFL.message import Message
from MiniFL.utils import get_num_bits

from .interfaces import Compressor, ContractiveCompressor, InputVarianceCompressor, UnbiasedCompressor


class IdentityCompressor(UnbiasedCompressor):
    def __init__(self, size: int):
        super().__init__(size)

    def compress(self, x: FloatTensor) -> Message:
        return Message(
            data=(x,),
            size=x.numel() * get_num_bits(x.dtype),
        )

    def decompress(self, msg: Message) -> FloatTensor:
        return msg.data[0]

    def omega(self) -> float:
        return 0


class TopKCompressor(ContractiveCompressor):
    def __init__(self, size: int, k: int = None, p: float = None):
        super().__init__(size)

        assert (k is None) ^ (p is None), "Either k or p must be specified"
        if k is None:
            self.k = math.ceil(p * size)
        else:
            self.k = k

    def compress(self, x: FloatTensor) -> Message:
        _, indexes = torch.topk(torch.abs(x), k=self.k, sorted=False)
        values = x[indexes]

        return Message(
            data=(indexes, values),
            size=values.numel() * get_num_bits(values.dtype)
            + min(self.k * math.log2(x.numel()), (x.numel() - self.k) * math.log2(x.numel()), x.numel()),
            metadata={"shape": x.shape},
        )

    def decompress(self, msg: Message) -> FloatTensor:
        indexes, values = msg.data
        x = torch.zeros(msg.metadata["shape"], dtype=values.dtype, device=values.device)
        x[indexes] = values
        return x

    def alpha(self) -> float:
        return self.k / self.size


class RandKBaseCompressor(Compressor):
    def __init__(self, size: int, k: int = None, p: float = None, seed=0):
        super().__init__(size)

        assert (k is None) ^ (p is None), "Either k or p must be specified"
        if k is None:
            self.k = math.ceil(p * size)
        else:
            self.k = k

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def compress(self, x: FloatTensor) -> Message:
        indexes = torch.randperm(x.numel(), generator=self.generator)[: self.k]
        values = x[indexes]

        return Message(
            data=(indexes, values),
            size=self.k * get_num_bits(values.dtype),
            metadata={"shape": x.shape},
        )

    def decompress(self, msg: Message) -> FloatTensor:
        indexes, values = msg.data
        x = torch.zeros(msg.metadata["shape"], dtype=values.dtype, device=values.device)
        x[indexes] = values
        return x


class RandKUnbiasedCompressor(RandKBaseCompressor, UnbiasedCompressor):
    def compress(self, x: FloatTensor) -> Message:
        msg = super().compress(x=x)
        scale = self.size / self.k
        msg.data = (msg.data[0], msg.data[1] * scale)
        return msg

    def omega(self) -> float:
        return self.size / self.k - 1


class RandKContractiveCompressor(RandKBaseCompressor, ContractiveCompressor):
    def alpha(self) -> float:
        return self.k / self.size


class PermKUnbiasedCompressor(UnbiasedCompressor, InputVarianceCompressor):
    def __init__(self, size: int, rank: int, world_size: int, seed=0):
        super().__init__(size=size)

        self.rank = rank
        self.world_size = world_size
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def compress(self, x: FloatTensor) -> Message:
        partition_id = torch.randperm(self.world_size, generator=self.generator)[self.rank]
        indexes = torch.tensor_split(torch.randperm(x.numel(), generator=self.generator), self.world_size)[partition_id]
        values = x[indexes] * self.world_size

        return Message(
            data=(indexes, values),
            size=values.numel() * get_num_bits(values.dtype),
            metadata={"shape": x.shape},
        )

    def decompress(self, msg: Message) -> FloatTensor:
        indexes, values = msg.data
        x = torch.zeros(msg.metadata["shape"], dtype=values.dtype, device=values.device)
        x[indexes] = values
        return x

    def ab(self) -> (float, float):
        if self.size >= self.world_size:
            return 1, 1
        else:
            a = b = 1 - (self.world_size - self.size) / (self.world_size - 1)
            return a, b

    def omega(self) -> float:
        return self.world_size - 1
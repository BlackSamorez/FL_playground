import math

import torch
from torch import FloatTensor, Tensor

from MiniFL.message import Message
from MiniFL.utils import get_num_bits

from .interfaces import Compressor, UnbiasedCompressor


class IdentityCompressor(UnbiasedCompressor):
    def __init__(self):
        pass

    def compress(self, x: FloatTensor) -> Message:
        return Message(
            data=(x,),
            size=x.numel() * get_num_bits(x.dtype),
        )

    def decompress(self, msg: Message) -> FloatTensor:
        return msg.data[0]


class TopPBiasedCompressor(Compressor):
    def __init__(self, p: float):
        self.p = p

    def compress(self, x: FloatTensor) -> Message:
        k = math.ceil(self.p * x.numel())
        _, indexes = torch.topk(torch.abs(x), k=k, sorted=False)
        values = x[indexes]

        return Message(
            data=(indexes, values),
            size=values.numel() * get_num_bits(values.dtype)
            + min(k * math.log2(x.numel()), (x.numel() - k) * math.log2(x.numel()), x.numel()),
            metadata={"shape": x.shape},
        )

    def decompress(self, msg: Message) -> FloatTensor:
        indexes, values = msg.data
        x = torch.zeros(msg.metadata["shape"], dtype=values.dtype, device=values.device)
        x[indexes] = values
        return x


class RandPBiasedCompressor(Compressor):
    def __init__(self, p: float, seed=0):
        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def compress(self, x: FloatTensor) -> Message:
        k = math.ceil(self.p * x.numel())
        indexes = torch.randperm(x.numel(), generator=self.generator)[:k]
        values = x[indexes]

        return Message(
            data=(indexes, values),
            size=k * get_num_bits(values.dtype),
            metadata={"shape": x.shape},
        )

    def decompress(self, msg: Message) -> FloatTensor:
        indexes, values = msg.data
        x = torch.zeros(msg.metadata["shape"], dtype=values.dtype, device=values.device)
        x[indexes] = values
        return x


class RandPUnbiasedCompressor(RandPBiasedCompressor, UnbiasedCompressor):
    def __init__(self, p: float):
        super().__init__(p=p)

    def compress(self, x: FloatTensor) -> Message:
        msg = super().compress(x=x)
        scale = x.numel() / math.ceil(self.p * x.numel())
        msg.data = (msg.data[0], msg.data[1] * scale)
        return msg


class PermKUnbiasedCompressor(UnbiasedCompressor):
    def __init__(self, rank: int, world_size: int, seed=0):
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

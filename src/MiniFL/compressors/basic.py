import math

import torch
from torch import FloatTensor, Tensor

from MiniFL.message import Message
from MiniFL.utils import get_num_bits

from .interfaces import Compressor, UnbiasedCompressor


class IdentityCompressor(Compressor):
    def __init__(self):
        pass

    def compress(self, x: FloatTensor) -> Message:
        return Message(
            data=(x,),
            size=x.numel() * get_num_bits(x.dtype),
        )

    def decompress(self, msg: Message) -> FloatTensor:
        return msg.data[0]


class TopKBiasedCompressor(Compressor):
    def __init__(self, k: int):
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


class RandKBiasedCompressor(Compressor):
    def __init__(self, k: int, seed=0):
        self.k = k
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def compress(self, x: FloatTensor) -> Message:
        indexes = torch.randperm(x.numel(), generator=self.generator)[: self.k]
        values = x[indexes]

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


class RandKUnbiasedCompressor(RandKBiasedCompressor, UnbiasedCompressor):
    def __init__(self, k: int):
        super().__init__(k=k)

    def compress(self, x: FloatTensor) -> Message:
        n = x.numel()
        msg = super().compress(x=x)
        msg.data = (msg.data[0], msg.data[1] * (n / self.k))
        return msg

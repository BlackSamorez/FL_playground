import math
from typing import Collection

import torch
from torch import Tensor

from MiniFL.message import Message
from MiniFL.utils import get_num_bits

from .interfaces import Compressor


class IdentityCompressor(Compressor):
    def __init__(self):
        pass

    def compress(self, data: Collection[Tensor]) -> Message:
        return Message(
            data=data,
            size=sum(tensor.numel() * get_num_bits(tensor.dtype) for tensor in data),
        )

    def decompress(self, msg: Message) -> Collection[Tensor]:
        return msg.data


class TopKBiasedCompressor(Compressor):
    def __init__(self, k: int):
        self.k = k

    def compress(self, data: Collection[Tensor]) -> Message:
        assert len(data) == 1
        x = data[0]
        assert x.dtype.is_floating_point

        _, indexes = torch.topk(torch.abs(x.data), k=self.k, sorted=False)
        values = x[indexes]

        return Message(
            data=(indexes, values),
            size=values.numel() * get_num_bits(values.dtype)
            + min(self.k * math.log2(x.numel()), (x.numel() - self.k) * math.log2(x.numel()), x.numel()),
            metadata={"shape": x.shape},
        )

    def decompress(self, msg: Message) -> Collection[Tensor]:
        indexes, values = msg.data
        x = torch.zeros(msg.metadata["shape"], dtype=values.dtype, device=values.device)
        x[indexes] = values
        return (x,)


class RandKCompressor(Compressor):
    def __init__(self, k: int, seed=0):
        self.k = k
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def compress(self, data: Collection[Tensor]) -> Message:
        assert len(data) == 1
        x = data[0]

        indexes = torch.randperm(x.numel(), generator=self.generator)[: self.k]
        values = x[indexes]

        return Message(
            data=(indexes, values),
            size=values.numel() * get_num_bits(values.dtype),
            metadata={"shape": x.shape},
        )

    def decompress(self, msg: Message) -> Collection[Tensor]:
        indexes, values = msg.data
        x = torch.zeros(msg.metadata["shape"], dtype=values.dtype, device=values.device)
        x[indexes] = values
        return (x,)

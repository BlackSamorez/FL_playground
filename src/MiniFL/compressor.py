import math
from abc import ABC, abstractmethod
from typing import Collection, Mapping

import torch
from torch import Tensor, nn

from MiniFL.message import Message

from .message import Message
from .utils import get_num_bits


class Flattener:
    def __init__(self, shapes: Mapping[str, torch.Size]) -> None:
        self.shapes = shapes

    def flatten(self, tensors: Mapping[str, Tensor]) -> Collection[Tensor]:
        return (torch.cat(tuple(tensors[name].flatten() for name in sorted(tensors))),)

    def unflatten(self, data: Collection[Tensor]) -> Mapping[str, Tensor]:
        assert len(data) == 1
        x = data[0]
        restored_tensors = {}
        for name in sorted(self.shapes):
            shape = self.shapes[name]
            restored_tensors[name] = x[: shape.numel()].view(*shape)
            x = x[shape.numel() :]
        return restored_tensors


class Compressor(ABC):
    @abstractmethod
    def compress(self, data: Collection[Tensor]) -> Message:
        pass

    @abstractmethod
    def decompress(self, msg: Message) -> Collection[Tensor]:
        pass


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
        masks = torch.zeros_like(x, dtype=torch.uint8)
        masks[indexes] = 1

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

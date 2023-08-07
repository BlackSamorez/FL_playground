from abc import ABC, abstractmethod
from typing import Collection, Mapping

import torch
from torch import Tensor, nn

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
    def compress(self, named_tensors: Collection[Tensor]) -> Message:
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

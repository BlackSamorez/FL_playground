from abc import ABC, abstractmethod
from typing import Collection, Mapping

import torch
from torch import Tensor, nn


class Compressor(ABC):
    @abstractmethod
    def compress(self, named_tensors: Mapping[str, Tensor]) -> Collection[Tensor]:
        pass

    @abstractmethod
    def decompress(self, data: Collection[Tensor]) -> Mapping[str, Tensor]:
        pass


class IdentityCompressor(Compressor):
    def __init__(self, model: nn.Module):
        self.shapes = {k: v.shape for k, v in model.named_parameters()}

    def compress(self, grad_dict: Mapping[str, Tensor]) -> Collection[Tensor]:
        return (
            torch.cat(tuple(grad_dict[name].flatten() for name in sorted(grad_dict))),
        )

    def decompress(self, data: Collection[Tensor]) -> Mapping[str, Tensor]:
        x = data[0]
        grad_dict = {}
        for name in sorted(self.shapes):
            shape = self.shapes[name]
            grad_dict[name] = x[: shape.numel()].view(*shape)
            x = x[shape.numel() :]
        return grad_dict

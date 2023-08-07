from typing import Collection, Mapping

import torch
from torch import FloatTensor, Tensor, nn


def get_grad_dict(module: nn.Module) -> Mapping[str, Tensor]:
    return {k: v.grad.detach() for k, v in module.named_parameters()}


def add_grad_dict(module: nn.Module, grad_dict: Mapping[str, Tensor]):
    for k, v in module.named_parameters():
        if v.grad is None:
            v.grad = grad_dict[k]
        else:
            v.grad += grad_dict[k]


def get_num_bits(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits
    else:
        return torch.iinfo(dtype).bits


class Flattener:
    def __init__(self, shapes: Mapping[str, torch.Size]) -> None:
        self.shapes = shapes

    def flatten(self, tensors: Mapping[str, FloatTensor]) -> FloatTensor:
        return torch.cat(tuple(tensors[name].clone().detach().flatten() for name in sorted(tensors)))

    def unflatten(self, x: FloatTensor) -> Mapping[str, FloatTensor]:
        restored_tensors = {}
        for name in sorted(self.shapes):
            shape = self.shapes[name]
            restored_tensors[name] = x[: shape.numel()].view(*shape)
            x = x[shape.numel() :]
        return restored_tensors

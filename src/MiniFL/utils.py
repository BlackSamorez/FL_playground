from typing import Mapping

import torch
from torch import Tensor, nn


def get_grad_dict(module: nn.Module) -> Mapping[str, Tensor]:
    return {k: v.grad.detach() for k, v in module.named_parameters()}


def add_grad_dict(module: nn.Module, grad_dict: Mapping[str, Tensor]):
    for k, v in module.named_parameters():
        v.grad = grad_dict[k]

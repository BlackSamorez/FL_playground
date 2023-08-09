from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import FloatTensor, Tensor, nn

from .utils import Flattener, add_grad_dict, get_grad_dict


class DifferentiableFn(ABC):
    @abstractmethod
    def is_full_grad(self) -> bool:
        pass

    @abstractmethod
    def get_value(self) -> float:
        pass

    @abstractmethod
    def get_parameters(self) -> FloatTensor:
        pass

    @abstractmethod
    def get_flat_grad_estimate() -> FloatTensor:
        pass

    @abstractmethod
    def step(delta: FloatTensor):
        pass

    @abstractmethod
    def zero_like_grad() -> FloatTensor:
        pass


class AutogradDifferentiableFn(DifferentiableFn):
    def __init__(self, fn, arg_tensor: FloatTensor):
        self.fn = fn
        self.arg_parameter = nn.Parameter(arg_tensor)
        assert self.arg_parameter.requires_grad

        self.optimizer = torch.optim.SGD([self.arg_parameter], lr=1)

    def is_full_grad(self) -> bool:
        return True

    def get_value(self) -> float:
        with torch.no_grad():
            return float(self.fn(self.arg_parameter.data))

    def get_parameters(self) -> FloatTensor:
        return self.arg_parameter.data.clone().detach()

    def get_flat_grad_estimate(self) -> FloatTensor:
        self.optimizer.zero_grad()
        self.fn(self.arg_parameter).backward()
        return self.arg_parameter.grad.data.clone().detach()

    def step(self, delta: FloatTensor):
        self.optimizer.zero_grad()
        self.arg_parameter.grad = -delta.clone().detach()
        self.optimizer.step()

    def zero_like_grad(self) -> FloatTensor:
        return torch.zeros_like(self.arg_parameter.data)


class NNDifferentiableFn(DifferentiableFn):
    def __init__(self, model: nn.Module, data: Tuple[Tensor, Tensor], loss_fn, batch_size: int, seed: int = 0):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1)
        self.flattener = Flattener(shapes={k: v.shape for k, v in self.model.named_parameters()})
        self.data = data
        self.loss_fn = loss_fn
        self.batch_size = batch_size

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def is_full_grad(self) -> bool:
        return self.batch_size == -1

    def get_value(self) -> float:
        with torch.no_grad():
            if self.batch_size == -1:
                return float(self.loss_fn(self.model(self.data[0]), self.data[1]))
            else:
                batch_idx = torch.randperm(self.data[1].shape[0], generator=self.generator)[: self.batch_size]
                return float(self.loss_fn(self.model(self.data[0][batch_idx]), self.data[1][batch_idx]))

    def get_parameters(self) -> FloatTensor:
        return self.flattener.flatten(self.model.state_dict())

    def get_flat_grad_estimate(self) -> FloatTensor:
        self.optimizer.zero_grad()
        if self.batch_size == -1:
            loss = self.loss_fn(self.model(self.data[0]), self.data[1])
        else:
            batch_idx = torch.randperm(self.data[1].shape[0], generator=self.generator)[: self.batch_size]
            loss = self.loss_fn(self.model(self.data[0][batch_idx]), self.data[1][batch_idx])
        loss.backward()

        return self.flattener.flatten(get_grad_dict(self.model))

    def step(self, delta: FloatTensor):
        self.optimizer.zero_grad()
        add_grad_dict(self.model, grad_dict=self.flattener.unflatten(-delta))  # torch minimizes by default
        self.optimizer.step()

    def zero_like_grad(self) -> FloatTensor:
        return torch.zeros_like(self.get_flat_grad_estimate())

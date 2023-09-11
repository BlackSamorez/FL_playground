from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cache
from typing import Collection, Tuple

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
    def get_flat_grad_estimate(self) -> FloatTensor:
        pass

    @abstractmethod
    def step(delta: FloatTensor):
        pass

    @abstractmethod
    def zero_like_grad(self) -> FloatTensor:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    def liptschitz_gradient_constant(self) -> float:
        raise NotImplementedError()

    @staticmethod
    def smoothness_variance(fns) -> float:
        raise NotImplementedError()


class MeanDifferentiableFn(DifferentiableFn):
    def __init__(self, fns: Collection[DifferentiableFn]):
        self.fns = deepcopy(fns)

    def is_full_grad(self) -> bool:
        return all(fn.is_full_grad() for fn in self.fns)

    def get_value(self) -> float:
        return sum(fn.get_value() for fn in self.fns) / len(self.fns)

    def get_parameters(self) -> FloatTensor:
        self.fns[0].get_parameters()

    def get_flat_grad_estimate(self) -> FloatTensor:
        return sum(fn.get_flat_grad_estimate() for fn in self.fns) / len(self.fns)

    def step(self, delta: FloatTensor):
        for fn in self.fns:
            fn.step(delta)

    def zero_like_grad(self) -> FloatTensor:
        return self.fns[0].zero_like_grad()

    def size(self) -> int:
        return self.fns[0].size()

    def liptschitz_gradient_constant(self) -> float:
        return sum(fn.liptschitz_gradient_constant() for fn in self.fns) / len(self.fns)


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

    def size(self) -> int:
        return self.arg_parameter.numel()


class NNDifferentiableFn(DifferentiableFn):
    def __init__(self, model: nn.Module, data: Tuple[Tensor, Tensor], loss_fn, batch_size: int, seed: int = 0):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1)
        self.flattener = Flattener(shapes={k: v.shape for k, v in self.model.named_parameters()})
        self.data = data
        self.loss_fn = loss_fn
        self.batch_size = batch_size

        if self.batch_size != -1:
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
        return torch.zeros_like(self.get_parameters())

    def size(self) -> int:
        return self.get_parameters().numel()


class LogisticRegression(NNDifferentiableFn):
    def __init__(self, data: Tuple[Tensor, Tensor], batch_size: int, weight: FloatTensor = None, seed: int = 0):
        if weight is None:
            weight = torch.zeros_like(data[0][[0]])
        model = nn.Linear(data[0].shape[1], 1, bias=False)
        model.weight.data = weight

        super().__init__(
            model=model,
            data=data,
            loss_fn=nn.BCEWithLogitsLoss(),
            batch_size=batch_size,
            seed=seed,
        )

    def liptschitz_gradient_constant(self):
        return self.data[0].square().sum(dim=1).max().item()


class ReplicatedFn(DifferentiableFn):
    def __init__(self, fn: DifferentiableFn, world_size: int):
        self.fn = fn

        self.world_size = world_size
        self.updates_accumulated = 0

        self.grad = self.fn.zero_like_grad()
        self.parameters = self.fn.zero_like_grad()
        self.value = 0
        self.needs_recalculation = True
        self.maybe_recalculate_()

    def maybe_recalculate_(self):
        if self.needs_recalculation:
            self.grad = self.fn.get_flat_grad_estimate()
            self.parameters = self.fn.get_parameters()
            self.value = self.fn.get_value()
            self.needs_recalculation = False

    def is_full_grad(self) -> bool:
        return self.fn.is_full_grad()

    def get_value(self) -> float:
        self.maybe_recalculate_()
        return self.value

    def get_parameters(self) -> FloatTensor:
        self.maybe_recalculate_()
        return self.parametersss

    def get_flat_grad_estimate(self) -> FloatTensor:
        self.maybe_recalculate_()
        return self.fn.get_flat_grad_estimate()

    def step(self, delta: FloatTensor):
        if self.updates_accumulated == 0:
            self.fn.step(delta)
            self.needs_recalculation = True

        self.updates_accumulated = (self.updates_accumulated + 1) % self.world_size

    def zero_like_grad(self) -> FloatTensor:
        return self.fn.zero_like_grad()

    def size(self) -> int:
        return self.fn.size()

    @cache
    def liptschitz_gradient_constant(self) -> float:
        return self.fn.liptschitz_gradient_constant()

    @staticmethod
    def smoothness_variance(fns: Collection[DifferentiableFn]) -> float:
        return 0

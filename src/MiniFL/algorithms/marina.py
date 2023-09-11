import math
from typing import Collection, Tuple

import torch
from scipy.optimize import minimize
from torch import FloatTensor

from MiniFL.compressors import Compressor, IdentityCompressor, PermKUnbiasedCompressor
from MiniFL.compressors.interfaces import InputVarianceCompressor
from MiniFL.fn import DifferentiableFn
from MiniFL.message import Message
from MiniFL.metrics import ClientStepMetrics, MasterStepMetrics

from .interfaces import Client, Master


def get_c(generator: torch.Generator, p: float) -> bool:
    return bool(torch.bernoulli(torch.Tensor([p]), generator=generator).item())


class MarinaClient(Client):
    def __init__(
        self,
        # Task
        fn: DifferentiableFn,
        # Communications
        uplink_compressor: Compressor,
        # Hyperparameters
        gamma: float,
        p: float,
        seed: int = 0,
    ):
        super().__init__(fn=fn)

        self.uplink_compressor = uplink_compressor
        self.identity_uplink_compressor = IdentityCompressor(fn.size())

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.p = p
        self.gamma = gamma

        self.grad_prev = None

    def step(self, broadcasted_master_tensor: FloatTensor) -> (Message, FloatTensor, ClientStepMetrics):
        self.fn.step(-broadcasted_master_tensor * self.gamma)
        # Construct and send g_i^{k+1}
        flattened_grad = self.fn.get_flat_grad_estimate()
        c = get_c(self.generator, self.p)
        if c or self.step_num == 0:  # always send full grad on first step
            msg = self.identity_uplink_compressor.compress(flattened_grad)
        else:
            msg = self.uplink_compressor.compress(flattened_grad - self.grad_prev)
        self.grad_prev = flattened_grad

        self.step_num += 1
        return (
            msg,
            flattened_grad,
            ClientStepMetrics(
                step=self.step_num,
                value=self.fn.get_value(),
                grad_norm=torch.linalg.vector_norm(flattened_grad),
            ),
        )


class MarinaMaster(Master):
    def __init__(
        self,
        # Task
        size: int,
        num_clients: int,
        # Hyperparameters
        gamma: float,
        p: float,
        seed: int = 0,
    ):
        super().__init__(size=size, num_clients=num_clients)
        self.downlink_compressor = IdentityCompressor(size)

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.p = p
        self.gamma = gamma

        self.g_prev = torch.zeros(size)

    def step(self, sum_worker_tensor: FloatTensor = None) -> Message:
        # g_{k+1} = \sum_{i=1}^n g_i^{k+1}
        c = get_c(self.generator, self.p)
        if c or self.step_num == 0:  # always receive full grad on first step
            self.g_prev = sum_worker_tensor / self.num_clients
        else:
            self.g_prev += sum_worker_tensor / self.num_clients

        self.step_num += 1
        return self.downlink_compressor.compress(self.g_prev)


def get_optimal_homogeneous_p(a, b, size):
    assert b == 0, "Not implemented"
    c = (32 + size) / (32 * size)

    def func_(p):
        return (p + (1 - p) * c) * (1 + math.sqrt((1 - p) / p * a))

    def grad_(p):
        return ((-a / p - a * (1 - p) / p**2) * (p + c * (1 - p))) / (2 * math.sqrt(a * (1 - p) / p)) + (1 - c) * (
            1 + math.sqrt(a * (1 - p) / p)
        )

    result = minimize(func_, 0.5, method="SLSQP", jac=grad_, bounds=[(1e-5, 1 - 1e-5)], tol=1e-10)
    assert result.success, result.message
    return result.x[0]


def get_theoretical_step_size_ab(a, b, client_fns, num_clients, p):
    liptschitz_constants = [fn.liptschitz_gradient_constant() for fn in client_fns]
    mean_liptschitz_gradient_constant = sum(liptschitz_constants) / num_clients
    mean_square_liptschitz_gradient_constant = (sum(l**2 for l in liptschitz_constants) / num_clients) ** (1 / 2)
    smoothness_variance = client_fns[0].smoothness_variance(client_fns)
    assert smoothness_variance <= mean_square_liptschitz_gradient_constant**2
    m = mean_liptschitz_gradient_constant + math.sqrt(
        ((1 - p) / p) * ((a - b) * mean_square_liptschitz_gradient_constant**2 + b * smoothness_variance**2)
    )
    return 1 / m


def get_marina_master_and_clients(
    client_fns: Collection[DifferentiableFn],
    compressors: Collection[Compressor],
    p: float = None,
    gamma: float = None,
    gamma_multiplier: float = None,
    seed: int = 0,
) -> Tuple[MarinaMaster, Collection[MarinaClient]]:
    num_clients = len(client_fns)
    size = client_fns[0].size()

    if isinstance(compressors[0], InputVarianceCompressor):
        a, b = compressors[0].ab()
    else:
        a, b = compressors[0].omega(), 0

    if p is None:
        p = get_optimal_homogeneous_p(a, b, size)

    if gamma is None:
        if gamma_multiplier is None:
            raise ValueError("Either gamma or gamma_multiplier must be specified")
        gamma = get_theoretical_step_size_ab(a, b, client_fns, num_clients, p)
        gamma *= gamma_multiplier

    master = MarinaMaster(
        size=size,
        num_clients=num_clients,
        gamma=gamma,
        p=p,
        seed=seed,
    )

    clients = [
        MarinaClient(
            fn=client_fns[i],
            uplink_compressor=compressors[i],
            gamma=gamma,
            p=p,
            seed=seed,
        )
        for i in range(num_clients)
    ]

    return master, clients


def get_permk_marina_master_and_clients(
    client_fns: Collection[DifferentiableFn],
    p: float,
    gamma: float = None,
    gamma_multiplier: float = None,
    compressors_seed: int = 0,
    seed: int = 0,
) -> Tuple[MarinaMaster, Collection[MarinaClient]]:
    num_clients = len(client_fns)
    size = client_fns[0].size()

    compressors = [
        PermKUnbiasedCompressor(size, rank=i, world_size=len(client_fns), seed=compressors_seed)
        for i in range(num_clients)
    ]

    return get_marina_master_and_clients(
        client_fns=client_fns,
        compressors=compressors,
        p=p,
        gamma=gamma,
        gamma_multiplier=gamma_multiplier,
        seed=seed,
    )

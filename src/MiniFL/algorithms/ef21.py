import math
from typing import Collection, Tuple

import torch
from torch import FloatTensor

from MiniFL.compressors import Compressor, IdentityCompressor
from MiniFL.fn import DifferentiableFn
from MiniFL.message import Message
from MiniFL.metrics import ClientStepMetrics, MasterStepMetrics

from .interfaces import Client, Master


class Ef21Client(Client):
    def __init__(
        self,
        # Task
        fn: DifferentiableFn,
        # Communications
        uplink_compressor: Compressor,
        # Hyperparameters
        gamma: float,
    ):
        super().__init__(fn=fn)

        self.uplink_compressor = uplink_compressor

        self.gamma = gamma

        self.g = self.fn.zero_like_grad()

    def prepare(self):
        pass

    def step(self, broadcasted_master_tensor: FloatTensor) -> (Message, ClientStepMetrics):
        self.fn.step(-broadcasted_master_tensor * self.gamma)

        grad_estimate = self.fn.get_flat_grad_estimate()
        msg = self.uplink_compressor.compress(grad_estimate - self.g)
        self.g = self.g + msg.data

        grad_estimate = self.fn.get_flat_grad_estimate()
        self.step_num += 1
        return msg, ClientStepMetrics(
            step=self.step_num - 1,
            value=self.fn.get_value(),
            grad_norm=torch.linalg.vector_norm(grad_estimate),
        )


class Ef21Master(Master):
    def __init__(
        self,
        # Task
        fn: DifferentiableFn,
        num_clients: int,
        # Hyperparameters
        gamma: float,
    ):
        super().__init__(fn=fn, num_clients=num_clients)
        self.gamma = gamma
        self.compressor = IdentityCompressor(fn.size())

        self.g = self.fn.zero_like_grad()

    def prepare(self):
        pass

    def step(self, sum_worker_tensor: FloatTensor) -> (Message, MasterStepMetrics):
        value = self.fn.get_value()
        self.g = self.g + sum_worker_tensor / self.num_clients
        self.fn.step(-self.g * self.gamma)
        msg = self.compressor.compress(self.g)

        self.step_num += 1
        return msg, MasterStepMetrics(
            step=self.step_num - 1,
            value=value,
            grad_norm=torch.linalg.vector_norm(self.g).item(),
        )


def get_ef21_master_and_clients(
    master_fn: DifferentiableFn,
    client_fns: Collection[DifferentiableFn],
    compressors: Collection[Compressor],
    gamma: float = None,
    gamma_multiplier: float = None,
) -> Tuple[Ef21Master, Collection[Ef21Client]]:
    num_clients = len(client_fns)
    if gamma is None:
        if gamma_multiplier is None:
            raise ValueError("Either gamma or gamma_multiplier must be specified")
        liptschitz_constants = [fn.liptschitz_gradient_constant() for fn in client_fns]
        mean_liptschitz_gradient_constant = sum(liptschitz_constants) / num_clients
        mean_square_liptschitz_gradient_constant = (sum(l**2 for l in liptschitz_constants) / num_clients) ** (1 / 2)
        alpha = compressors[0].alpha()
        gamma = gamma_multiplier / (
            mean_liptschitz_gradient_constant
            + mean_square_liptschitz_gradient_constant * math.sqrt((1 - alpha) / (1 - math.sqrt(1 - alpha)) ** 2)
        )

    master = Ef21Master(
        fn=master_fn,
        num_clients=num_clients,
        gamma=gamma,
    )

    clients = [
        Ef21Client(
            fn=client_fns[i],
            uplink_compressor=compressors[i],
            gamma=gamma,
        )
        for i in range(num_clients)
    ]

    return master, clients

from typing import Collection, Tuple

import torch
from torch import FloatTensor

from MiniFL.compressors import IdentityCompressor
from MiniFL.fn import DifferentiableFn
from MiniFL.message import Message
from MiniFL.metrics import ClientStepMetrics, MasterStepMetrics

from .interfaces import Client, Master


class GDClient(Client):
    def __init__(
        self,
        fn: DifferentiableFn,
        gamma: float,
    ):
        super().__init__(fn=fn)
        self.compressor = IdentityCompressor(fn.size())
        self.gamma = gamma

    def step(self, broadcasted_master_tensor: FloatTensor) -> (Message, FloatTensor, ClientStepMetrics):
        self.fn.step(-broadcasted_master_tensor * self.gamma)
        grad_estimate = self.fn.get_flat_grad_estimate()
        self.step_num += 1
        return (
            self.compressor.compress(grad_estimate),
            grad_estimate,
            ClientStepMetrics(
                step=self.step_num - 1,
                value=self.fn.get_value(),
                grad_norm=torch.linalg.vector_norm(grad_estimate),
            ),
        )


class GDMaster(Master):
    def __init__(
        self,
        size: int,
        num_clients: int,
        gamma: float,
    ):
        super().__init__(size=size, num_clients=num_clients)
        self.gamma = gamma
        self.compressor = IdentityCompressor(size)

    def step(self, sum_worker_tensor: FloatTensor) -> Message:
        global_grad_estimate = sum_worker_tensor / self.num_clients
        self.step_num += 1
        return self.compressor.compress(global_grad_estimate)


def get_gd_master_and_clients(
    client_fns: Collection[DifferentiableFn],
    gamma: float = None,
    gamma_multiplier: float = None,
) -> Tuple[GDMaster, Collection[GDClient]]:
    num_clients = len(client_fns)
    size = client_fns[0].size()
    if gamma is None:
        if gamma_multiplier is None:
            raise ValueError("Either gamma or gamma_multiplier must be specified")
        mean_liptschitz_gradient_constant = sum(fn.liptschitz_gradient_constant() for fn in client_fns) / num_clients
        gamma = gamma_multiplier / mean_liptschitz_gradient_constant

    master = GDMaster(
        size=size,
        num_clients=num_clients,
        gamma=gamma,
    )

    clients = [
        GDClient(
            fn=client_fns[i],
            gamma=gamma,
        )
        for i in range(num_clients)
    ]

    return master, clients

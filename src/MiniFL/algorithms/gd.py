from typing import Collection, Tuple

import torch

from MiniFL.communications import DataReceiver, DataSender, get_sender_receiver
from MiniFL.compressors import Compressor, IdentityCompressor
from MiniFL.fn import DifferentiableFn
from MiniFL.metrics import ClientStepMetrics, MasterStepMetrics

from .interfaces import Client, Master


class GDClient(Client):
    def __init__(
        self,
        fn: DifferentiableFn,
        data_sender: DataSender,
        data_receiver: DataReceiver,
        gamma: float,
    ):
        super().__init__(fn=fn)

        self.data_sender = data_sender
        self.data_receiver = data_receiver
        self.compressor = IdentityCompressor(fn.size())

        self.gamma = gamma

    def prepare(self):
        pass

    async def step(self) -> ClientStepMetrics:
        loss, grad_norm = await self.send_grad_get_loss_()
        await self.apply_global_step_()

        self.step_num += 1
        return ClientStepMetrics(
            step=self.step_num - 1,
            value=loss,
            total_bits_sent=self.data_sender.n_bits_passed,
            total_bits_received=self.data_receiver.n_bits_passed,
            grad_norm=grad_norm,
        )

    async def send_grad_get_loss_(self) -> (float, float):
        flat_grad_estimate = self.fn.get_flat_grad_estimate()
        msg = self.compressor.compress(flat_grad_estimate)
        await self.data_sender.send(msg)
        return self.fn.get_value(), torch.linalg.vector_norm(flat_grad_estimate)

    async def apply_global_step_(self):
        msg = await self.data_receiver.recv()
        aggregated_grad_estimate = self.compressor.decompress(msg)
        self.fn.step(-aggregated_grad_estimate * self.gamma)


class GDMaster(Master):
    def __init__(
        self,
        fn: DifferentiableFn,
        data_senders: Collection[DataSender],
        data_receivers: Collection[DataReceiver],
        gamma: float,
    ):
        super().__init__(fn=fn)

        self.data_senders = data_senders
        self.data_receivers = data_receivers
        self.compressor = IdentityCompressor(fn.size())

        self.gamma = gamma

    def prepare(self):
        pass

    async def step(self) -> MasterStepMetrics:
        aggregated_gradients = self.fn.zero_like_grad()
        for receiver in self.data_receivers:
            msg = await receiver.recv()
            aggregated_gradients += self.compressor.decompress(msg)
        aggregated_gradients /= len(self.data_receivers)

        for sender in self.data_senders:
            msg = self.compressor.compress(aggregated_gradients)
            await sender.send(msg)

        self.fn.step(-aggregated_gradients * self.gamma)

        self.step_num += 1
        return MasterStepMetrics(
            step=self.step_num - 1,
            value=self.fn.get_value(),
            total_bits_sent=sum(s.n_bits_passed for s in self.data_senders),
            total_bits_received=sum(r.n_bits_passed for r in self.data_receivers),
            grad_norm=torch.linalg.vector_norm(aggregated_gradients).item(),
        )


def get_gd_master_and_clients(
    master_fn: DifferentiableFn,
    client_fns: Collection[DifferentiableFn],
    gamma: float = None,
    gamma_multiplier: float = None,
) -> Tuple[GDMaster, Collection[GDClient]]:
    num_clients = len(client_fns)
    if gamma is None:
        if gamma_multiplier is None:
            raise ValueError("Either gamma or gamma_multiplier must be specified")
        mean_liptschitz_gradient_constant = sum(fn.liptschitz_gradient_constant() for fn in client_fns) / num_clients
        gamma = gamma_multiplier / mean_liptschitz_gradient_constant

    uplink_comms = [get_sender_receiver() for _ in range(num_clients)]
    downlink_comms = [get_sender_receiver() for _ in range(num_clients)]

    master = GDMaster(
        fn=master_fn,
        data_senders=[s for s, r in downlink_comms],
        data_receivers=[r for s, r in uplink_comms],
        gamma=gamma,
    )

    clients = [
        GDClient(
            fn=client_fns[i],
            data_sender=uplink_comms[i][0],
            data_receiver=downlink_comms[i][1],
            gamma=gamma,
        )
        for i in range(num_clients)
    ]

    return master, clients

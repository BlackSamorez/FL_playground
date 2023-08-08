from typing import Collection, Tuple

import torch

from MiniFL.communications import DataReceiver, DataSender, get_sender_receiver
from MiniFL.compressors import Compressor, IdentityCompressor
from MiniFL.fn import DifferentiableFn

from .interfaces import Client, Master


class GDClient(Client):
    def __init__(
        self,
        fn: DifferentiableFn,
        data_sender: DataSender,
        data_receiver: DataReceiver,
        gamma: float,
    ):
        self.fn = fn

        self.data_sender = data_sender
        self.data_receiver = data_receiver
        self.compressor = IdentityCompressor()

        self.gamma = gamma

    def prepare(self):
        pass

    def step(self) -> float:
        loss = self.send_grad_get_loss_()
        self.apply_global_step_()
        return loss

    def send_grad_get_loss_(self) -> float:
        flat_grad_estimate = self.fn.get_flat_grad_estimate()
        msg = self.compressor.compress(flat_grad_estimate)
        self.data_sender.send(msg)
        return self.fn.get_value()

    def apply_global_step_(self):
        msg = self.data_receiver.recv()
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
        self.fn = fn
        self.data_senders = data_senders
        self.data_receivers = data_receivers
        self.compressor = IdentityCompressor()

        self.gamma = gamma

    def prepare(self):
        pass

    def step(self) -> float:
        aggregated_gradients = self.fn.zero_like_grad()
        for receiver in self.data_receivers:
            msg = receiver.recv()
            aggregated_gradients += self.compressor.decompress(msg)
        aggregated_gradients /= len(self.data_receivers)

        for sender in self.data_senders:
            msg = self.compressor.compress(aggregated_gradients)
            sender.send(msg)

        self.fn.step(-aggregated_gradients * self.gamma)

        with torch.no_grad():
            return self.fn.get_value()


def get_gd_master_and_clients(
    master_fn: DifferentiableFn,
    client_fns: Collection[DifferentiableFn],
    gamma: float,
) -> Tuple[GDMaster, Collection[GDClient]]:
    num_clients = len(client_fns)

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

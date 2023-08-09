from typing import Collection, Tuple

import torch
from torch import FloatTensor

from MiniFL.communications import DataReceiver, DataSender, get_sender_receiver
from MiniFL.compressors import CocktailCompressor, Compressor
from MiniFL.fn import DifferentiableFn
from MiniFL.metrics import ClientStepMetrics, MasterStepMetrics

from .interfaces import Client, Master


def get_c(generator: torch.Generator, p: float) -> bool:
    return bool(torch.bernoulli(torch.Tensor([p]), generator=generator).item())


class CocktailGDClient(Client):
    def __init__(
        self,
        # Task
        fn: DifferentiableFn,
        # Communications
        data_sender: DataSender,
        data_receiver: DataReceiver,
        uplink_compressor: Compressor,
        downlink_compressor: Compressor,
        # Hyperparameters
        gamma: float,
    ):
        super().__init__(fn=fn)

        self.data_sender = data_sender
        self.data_receiver = data_receiver
        self.uplink_compressor = uplink_compressor
        self.downlink_compressor = downlink_compressor

        self.gamma = gamma

        self.global_parameters = self.fn.get_parameters()

    def prepare(self):
        pass

    def step(self) -> float:
        value = self.fn.get_value()
        grad_estimate = self.compute_thread_()
        compressed_delta, compressed_global_delta = self.communication_thread_()
        self.apply_updates_(grad_estimate, compressed_delta, compressed_global_delta)

        self.step_num += 1
        return ClientStepMetrics(
            step=self.step_num - 1,
            value=value,
            total_bits_sent=self.data_sender.n_bits_passed,
            total_bits_received=self.data_receiver.n_bits_passed,
            grad_norm=torch.linalg.vector_norm(grad_estimate),
        )

    def compute_thread_(self) -> FloatTensor:
        return self.fn.get_flat_grad_estimate()

    def communication_thread_(self) -> (FloatTensor, FloatTensor):
        # \delta_t^{(i)} = x_t^{(i)} - x'_{t}^{(i)}
        delta = self.fn.get_parameters() - self.global_parameters

        uplink_msg = self.uplink_compressor.compress(delta)
        self.data_sender.send(uplink_msg)
        downlink_msg = self.data_receiver.recv()

        return self.uplink_compressor.decompress(uplink_msg), self.downlink_compressor.decompress(downlink_msg)

    def apply_updates_(
        self, grad_estimate: FloatTensor, compressed_delta: FloatTensor, compressed_global_delta: FloatTensor
    ):
        # Set x_{t+1}^{(i)} = x_{t}^{(i)} - \gamma g_t^{(i)} + C[\Delta_t] - C[\delta_t^{(i)}]
        self.fn.step(-grad_estimate * self.gamma + compressed_global_delta - compressed_delta)
        # Set x'_{t+1}^{(i)} = x'_{t}^{(i)} + C[\Delta_t]
        self.global_parameters += compressed_global_delta


class CocktailGDMaster(Master):
    def __init__(
        self,
        # Task
        fn: DifferentiableFn,
        # Communications
        data_senders: Collection[DataSender],
        data_receivers: Collection[DataReceiver],
        uplink_compressors: Collection[Compressor],
        downlink_compressor: Collection[Compressor],
    ):
        super().__init__(fn=fn)

        self.data_senders = data_senders
        self.data_receivers = data_receivers
        self.uplink_compressors = uplink_compressors
        self.downlink_compressor = downlink_compressor

        self.e = self.fn.zero_like_grad()

    def prepare(self):
        pass

    def step(self) -> float:
        # Aggregate compressed \delta_t^{(i)} from all workers
        global_delta = self.e.clone().detach()
        for reciever, compressor in zip(self.data_receivers, self.uplink_compressors):
            msg = reciever.recv()
            global_delta += compressor.decompress(msg) / len(self.data_senders)

        # Broadcast compressed Delta_t to all workers
        msg = self.downlink_compressor.compress(global_delta)
        for sender in self.data_senders:
            sender.send(msg)

        # Update e_{t+1}
        compressed_global_delta = self.downlink_compressor.decompress(msg)
        self.e = global_delta - compressed_global_delta

        # Update global model
        self.fn.step(compressed_global_delta)

        self.step_num += 1
        return MasterStepMetrics(
            step=self.step_num - 1,
            value=self.fn.get_value(),
            total_bits_sent=sum(s.n_bits_passed for s in self.data_senders),
            total_bits_received=sum(r.n_bits_passed for r in self.data_receivers),
            grad_norm=torch.linalg.vector_norm(compressed_global_delta),
        )


def get_cocktailgd_master_and_clients(
    master_fn: DifferentiableFn,
    client_fns: Collection[DifferentiableFn],
    gamma: float,
    rand_p: float = 0.1,
    top_p: float = 0.2,
    bits: int = 4,
    seed: int = 0,
) -> Tuple[CocktailGDMaster, Collection[CocktailGDClient]]:
    num_clients = len(client_fns)

    uplink_comms = [get_sender_receiver() for _ in range(num_clients)]
    uplink_compressors = [
        CocktailCompressor(size=master_fn.size(), rand_p=rand_p, top_p=top_p, bits=bits, seed=seed + i)
        for i in range(num_clients)
    ]
    downlink_compressor = CocktailCompressor(
        size=master_fn.size(), rand_p=rand_p, top_p=top_p, bits=bits, seed=seed - 1
    )
    downlink_comms = [get_sender_receiver() for _ in range(num_clients)]

    master = CocktailGDMaster(
        fn=master_fn,
        data_senders=[s for s, r in downlink_comms],
        data_receivers=[r for s, r in uplink_comms],
        uplink_compressors=uplink_compressors,
        downlink_compressor=downlink_compressor,
    )

    clients = [
        CocktailGDClient(
            fn=client_fns[i],
            data_sender=uplink_comms[i][0],
            data_receiver=downlink_comms[i][1],
            uplink_compressor=uplink_compressors[i],
            downlink_compressor=downlink_compressor,
            gamma=gamma,
        )
        for i in range(num_clients)
    ]

    return master, clients

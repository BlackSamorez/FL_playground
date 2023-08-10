import math
from typing import Collection, Tuple

import torch

from MiniFL.communications import DataReceiver, DataSender, get_sender_receiver
from MiniFL.compressors import Compressor, IdentityCompressor, PermKUnbiasedCompressor
from MiniFL.fn import DifferentiableFn
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
        data_sender: DataSender,
        data_receiver: DataReceiver,
        uplink_compressor: Compressor,
        # Hyperparameters
        gamma: float,
        p: float,
        seed: int = 0,
    ):
        super().__init__(fn=fn)

        self.data_sender = data_sender
        self.data_receiver = data_receiver
        self.uplink_compressor = uplink_compressor
        self.identity_uplink_compressor = IdentityCompressor(fn.size())
        self.identity_downlink_compressor = IdentityCompressor(fn.size())

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.p = p
        self.gamma = gamma

        self.grad_prev = None

    async def prepare(self):
        # Init \nabla f_i(x^0)
        self.grad_prev = self.fn.get_flat_grad_estimate()
        # And send it to master
        await self.data_sender.send(self.identity_uplink_compressor.compress(self.grad_prev))

    async def step(self) -> float:
        # Receive g^k from master and apply it
        grad_norm = await self.apply_global_step_()
        # Construct and send g_i^{k+1}
        loss = await self.send_grad_get_loss_()

        self.step_num += 1
        return ClientStepMetrics(
            step=self.step_num - 1,
            value=loss,
            total_bits_sent=self.data_sender.n_bits_passed,
            total_bits_received=self.data_receiver.n_bits_passed,
            grad_norm=grad_norm,
        )

    async def send_grad_get_loss_(self) -> float:
        flattened_grad = self.fn.get_flat_grad_estimate()

        c = get_c(self.generator, self.p)
        if c:
            msg = self.identity_uplink_compressor.compress(flattened_grad)
        else:
            msg = self.uplink_compressor.compress(flattened_grad - self.grad_prev)
        await self.data_sender.send(msg)

        self.grad_prev = flattened_grad
        return self.fn.get_value()

    async def apply_global_step_(self) -> float:
        msg = await self.data_receiver.recv()
        aggregated_grad_estimate = self.identity_downlink_compressor.decompress(msg)
        self.fn.step(-aggregated_grad_estimate * self.gamma)
        return torch.linalg.vector_norm(aggregated_grad_estimate)


class MarinaMaster(Master):
    def __init__(
        self,
        # Task
        fn: DifferentiableFn,
        # Communications
        data_senders: Collection[DataSender],
        data_receivers: Collection[DataReceiver],
        uplink_compressors: Collection[Compressor],
        # Hyperparameters
        gamma: float,
        p: float,
        seed: int = 0,
    ):
        super().__init__(fn=fn)

        self.data_senders = data_senders
        self.data_receivers = data_receivers
        self.uplink_compressors = uplink_compressors
        self.identity_uplink_compressors = [IdentityCompressor(fn.size()) for _ in range(len(data_receivers))]
        self.downlink_compressors = [IdentityCompressor(fn.size()) for _ in range(len(data_senders))]

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.p = p
        self.gamma = gamma

        self.g_prev = self.fn.zero_like_grad()

    async def prepare(self):
        # Initialize g_0
        await self.process_full_grads_()

    async def step(self) -> float:
        # Broadcast g_t to all workers
        for sender, compressor in zip(self.data_senders, self.downlink_compressors):
            msg = compressor.compress(self.g_prev)
            await sender.send(msg)
        self.fn.step(-self.g_prev * self.gamma)
        grad_norm = torch.linalg.vector_norm(self.g_prev).item()

        # g_{k+1} = \sum_{i=1}^n g_i^{k+1}
        c = get_c(self.generator, self.p)
        if c:
            await self.process_full_grads_()
        else:
            await self.process_compressed_shifts_()

        self.step_num += 1
        return MasterStepMetrics(
            step=self.step_num - 1,
            value=self.fn.get_value(),
            total_bits_sent=sum(s.n_bits_passed for s in self.data_senders),
            total_bits_received=sum(r.n_bits_passed for r in self.data_receivers),
            grad_norm=grad_norm,
        )

    async def process_full_grads_(self):
        self.g_prev = self.fn.zero_like_grad()
        for reciever, compressor in zip(self.data_receivers, self.identity_uplink_compressors):
            msg = await reciever.recv()
            self.g_prev += compressor.decompress(msg)
        self.g_prev /= len(self.data_senders)

    async def process_compressed_shifts_(self):
        self.g_prev *= len(self.data_senders)
        for receiver, compressor in zip(self.data_receivers, self.uplink_compressors):
            msg = await receiver.recv()
            self.g_prev += compressor.decompress(msg)
        self.g_prev /= len(self.data_senders)


def get_marina_master_and_clients(
    master_fn: DifferentiableFn,
    client_fns: Collection[DifferentiableFn],
    compressors: Collection[Compressor],
    p: float,
    gamma: float = None,
    gamma_multiplier: float = None,
    seed: int = 0,
) -> Tuple[MarinaMaster, Collection[MarinaClient]]:
    num_clients = len(client_fns)
    if gamma is None:
        if gamma_multiplier is None:
            raise ValueError("Either gamma or gamma_multiplier must be specified")
        mean_square_liptschitz_gradient_constant = (
            sum(fn.liptschitz_gradient_constant() ** 2 for fn in client_fns) / num_clients
        ) ** (1 / 2)
        gamma = gamma_multiplier / mean_square_liptschitz_gradient_constant

    uplink_comms = [get_sender_receiver() for _ in range(num_clients)]
    downlink_comms = [get_sender_receiver() for _ in range(num_clients)]

    master = MarinaMaster(
        fn=master_fn,
        data_senders=[s for s, r in downlink_comms],
        data_receivers=[r for s, r in uplink_comms],
        uplink_compressors=compressors,
        gamma=gamma,
        p=p,
        seed=seed,
    )

    clients = [
        MarinaClient(
            fn=client_fns[i],
            data_sender=uplink_comms[i][0],
            data_receiver=downlink_comms[i][1],
            uplink_compressor=compressors[i],
            gamma=gamma,
            p=p,
            seed=seed,
        )
        for i in range(num_clients)
    ]

    return master, clients


def get_permk_marina_master_and_clients(
    master_fn: DifferentiableFn,
    client_fns: Collection[DifferentiableFn],
    p: float,
    gamma: float = None,
    gamma_multiplier: float = None,
    compressors_seed: int = 0,
    seed: int = 0,
) -> Tuple[MarinaMaster, Collection[MarinaClient]]:
    num_clients = len(client_fns)
    compressors = [
        PermKUnbiasedCompressor(master_fn.size, rank=i, world_size=len(client_fns), seed=compressors_seed)
        for i in range(len(client_fns))
    ]

    if gamma is None:
        if gamma_multiplier is None:
            raise ValueError("Either gamma or gamma_multiplier must be specified")
        liptschitz_constants = [fn.liptschitz_gradient_constant() for fn in client_fns]
        mean_liptschitz_gradient_constant = sum(liptschitz_constants) / num_clients
        mean_square_liptschitz_gradient_constant = (sum(l**2 for l in liptschitz_constants) / num_clients) ** (1 / 2)
        smoothness_variance = client_fns[0].smoothness_variance(client_fns)
        assert smoothness_variance <= mean_square_liptschitz_gradient_constant**2
        omega = compressors[0].omega()
        m = mean_liptschitz_gradient_constant + math.sqrt(
            ((1 - p) / p)
            * (
                ((omega + 1) / num_clients - 1) * mean_square_liptschitz_gradient_constant**2
                + smoothness_variance**2
            )
        )
        gamma = gamma_multiplier / m

    return get_marina_master_and_clients(
        master_fn=master_fn,
        client_fns=client_fns,
        compressors=compressors,
        p=p,
        gamma=gamma,
        gamma_multiplier=gamma_multiplier,
        seed=seed,
    )

from copy import deepcopy
from queue import SimpleQueue
from typing import Collection, Mapping, Tuple

import torch
from torch import Tensor, nn

from MiniFL.communications import DataReceiver, DataSender, get_sender_receiver
from MiniFL.compressors import Compressor, IdentityCompressor, PermKUnbiasedCompressor
from MiniFL.utils import Flattener, add_grad_dict, get_grad_dict


def get_c(generator: torch.Generator, p: float) -> bool:
    return bool(torch.bernoulli(torch.Tensor([p]), generator=generator).item())


class MarinaClient:
    def __init__(
        self,
        # Task
        data: Tuple[Tensor, Tensor],
        model: nn.Module,
        loss_fn,
        # Communications
        data_sender: DataSender,
        data_receiver: DataReceiver,
        uplink_compressor: Compressor,
        # Hyperparameters
        gamma: float,
        p: float,
        seed: int = 0,
    ):
        self.data = data
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=gamma)
        self.flattener = Flattener(shapes={k: v.shape for k, v in self.model.named_parameters()})
        self.loss_fn = loss_fn

        self.data_sender = data_sender
        self.data_receiver = data_receiver
        self.uplink_compressor = uplink_compressor
        self.identity_uplink_compressor = IdentityCompressor()
        self.identity_downlink_compressor = IdentityCompressor()

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.p = p

    def prepare(self):
        # Init \nabla f_i(x^0)
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(self.data[0]), self.data[1])
        loss.backward()
        self.flattened_prev_grad = self.flattener.flatten(get_grad_dict(self.model))
        # And send it to master
        self.data_sender.send(self.identity_uplink_compressor.compress(self.flattened_prev_grad))

    def step(self) -> float:
        # Receive g^k from master and apply it
        self.apply_global_step_()
        # Construct and send g_i^{k+1}
        return self.send_grad_get_loss_()

    def send_grad_get_loss_(self) -> float:
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(self.data[0]), self.data[1])
        loss.backward()

        grad_dict = get_grad_dict(self.model)
        flattened_grad = self.flattener.flatten(grad_dict)

        c = get_c(self.generator, self.p)
        if c:
            msg = self.identity_uplink_compressor.compress(flattened_grad)
        else:
            msg = self.uplink_compressor.compress(flattened_grad - self.flattened_prev_grad)
        self.data_sender.send(msg)

        self.flattened_prev_grad = flattened_grad
        return float(loss)

    def apply_global_step_(self):
        msg = self.data_receiver.recv()
        grad_dict = self.flattener.unflatten(self.identity_downlink_compressor.decompress(msg))

        self.optimizer.zero_grad()
        add_grad_dict(self.model, grad_dict=grad_dict)
        self.optimizer.step()


class MarinaMaster:
    def __init__(
        self,
        # Task
        eval_data: Tuple[Tensor, Tensor],
        model: nn.Module,
        loss_fn,
        # Communications
        data_senders: Collection[DataSender],
        data_receivers: Collection[DataReceiver],
        uplink_compressors: Collection[Compressor],
        # Hyperparameters
        gamma: float,
        p: float,
        seed: int = 0,
    ):
        self.eval_data = eval_data
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=gamma)
        self.flattener = Flattener(shapes={k: v.shape for k, v in self.model.named_parameters()})
        self.loss_fn = loss_fn

        self.data_senders = data_senders
        self.data_receivers = data_receivers
        self.uplink_compressors = uplink_compressors
        self.identity_uplink_compressors = [IdentityCompressor() for _ in range(len(data_receivers))]
        self.downlink_compressors = [IdentityCompressor() for _ in range(len(data_senders))]

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.p = p

    def prepare(self):
        # Initialize g_0
        self.process_full_grads_()

    def step(self) -> float:
        # Broadcast g_t to all workers
        for sender, compressor in zip(self.data_senders, self.downlink_compressors):
            grad_dict = get_grad_dict(self.model)
            msg = compressor.compress(self.flattener.flatten(grad_dict))
            sender.send(msg)
        self.optimizer.step()

        # g_{k+1} = g_k + \sum_{i=1}^n g_i^{k+1}
        c = get_c(self.generator, self.p)
        if c:
            self.process_full_grads_()
        else:
            self.process_compressed_shifts_()

        return self.val_loss_()

    def scale_grads_(self, scale: float):
        for v in self.model.parameters():
            v.grad *= scale

    def process_full_grads_(self):
        self.model.zero_grad()
        for reciever, compressor in zip(self.data_receivers, self.identity_uplink_compressors):
            msg = reciever.recv()
            grad_dict = self.flattener.unflatten(compressor.decompress(msg))
            add_grad_dict(self.model, grad_dict=grad_dict)
        self.scale_grads_(1 / len(self.data_senders))

    def process_compressed_shifts_(self):
        self.scale_grads_(len(self.data_senders))
        for receiver, compressor in zip(self.data_receivers, self.uplink_compressors):
            msg = receiver.recv()
            grad_dict = self.flattener.unflatten(compressor.decompress(msg))
            add_grad_dict(self.model, grad_dict=grad_dict)
        self.scale_grads_(1 / len(self.data_senders))

    def val_loss_(self) -> float:
        with torch.no_grad():
            return float(self.loss_fn(self.model(self.eval_data[0]), self.eval_data[1]))


def get_marina_master_and_clients(
    clients_data: Collection[Collection[Tuple[Tensor, Tensor]]],
    eval_data: Collection[Tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn,
    compressors: Collection[Compressor],
    gamma: float,
    p: float,
    seed: int = 0,
) -> Tuple[MarinaMaster, Collection[MarinaClient]]:
    num_clients = len(clients_data)

    uplink_comms = [get_sender_receiver() for _ in range(num_clients)]
    downlink_comms = [get_sender_receiver() for _ in range(num_clients)]
    client_models = [deepcopy(model) for _ in range(num_clients)]

    master = MarinaMaster(
        eval_data=eval_data,
        model=model,
        loss_fn=loss_fn,
        data_senders=[s for s, r in downlink_comms],
        data_receivers=[r for s, r in uplink_comms],
        uplink_compressors=compressors,
        gamma=gamma,
        p=p,
        seed=seed,
    )

    clients = []
    for i in range(num_clients):
        client = MarinaClient(
            data=clients_data[i],
            model=client_models[i],
            loss_fn=loss_fn,
            data_sender=uplink_comms[i][0],
            data_receiver=downlink_comms[i][1],
            uplink_compressor=compressors[i],
            gamma=gamma,
            p=p,
            seed=seed,
        )
        clients.append(client)

    return master, clients


def get_permk_marina_master_and_clients(
    clients_data: Collection[Collection[Tuple[Tensor, Tensor]]],
    eval_data: Collection[Tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn,
    gamma: float,
    p: float,
    compressors_seed: int = 0,
    seed: int = 0,
) -> Tuple[MarinaMaster, Collection[MarinaClient]]:
    return get_marina_master_and_clients(
        clients_data=clients_data,
        eval_data=eval_data,
        model=model,
        loss_fn=loss_fn,
        compressors=[
            PermKUnbiasedCompressor(rank=i, world_size=len(clients_data), seed=compressors_seed)
            for i in range(len(clients_data))
        ],
        gamma=gamma,
        p=p,
        seed=seed,
    )

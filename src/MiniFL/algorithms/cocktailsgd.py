from copy import deepcopy
from queue import SimpleQueue
from typing import Collection, Mapping, Tuple

import torch
from torch import FloatTensor, Tensor, nn

from MiniFL.communications import DataReceiver, DataSender, get_sender_receiver
from MiniFL.compressors import CocktailCompressor, Compressor
from MiniFL.utils import Flattener, add_grad_dict, get_grad_dict

from .interfaces import Client, Master


def get_c(generator: torch.Generator, p: float) -> bool:
    return bool(torch.bernoulli(torch.Tensor([p]), generator=generator).item())


class CocktailGDClient(Client):
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
        downlink_compressor: Compressor,
        # Hyperparameters
        gamma: float,
    ):
        self.data = data
        self.global_model = model
        self.global_optimizer = torch.optim.SGD(self.global_model.parameters(), lr=1)
        self.local_model = deepcopy(model)
        self.local_optimizer = torch.optim.SGD(self.local_model.parameters(), lr=1)
        self.flattener = Flattener(shapes={k: v.shape for k, v in self.local_model.named_parameters()})
        self.loss_fn = loss_fn

        self.data_sender = data_sender
        self.data_receiver = data_receiver
        self.uplink_compressor = uplink_compressor
        self.downlink_compressor = downlink_compressor

        self.gamma = gamma

    def prepare(self):
        pass

    def step(self) -> float:
        loss = self.compute_thread_()
        compressed_delta, compressed_global_delta = self.communication_thread_()
        self.apply_updates_(compressed_delta, compressed_global_delta)
        return loss

    def compute_thread_(self) -> float:
        self.local_optimizer.zero_grad()
        loss = self.loss_fn(self.local_model(self.data[0]), self.data[1]) * self.gamma
        loss.backward()
        return float(loss)

    def communication_thread_(self) -> (FloatTensor, FloatTensor):
        # \delta_t^{(i)} = x_t^{(i)} - x'_{t}^{(i)}
        delta = self.flattener.flatten(self.local_model.state_dict()) - self.flattener.flatten(
            self.global_model.state_dict()
        )

        uplink_msg = self.uplink_compressor.compress(delta)
        self.data_sender.send(uplink_msg)
        downlink_msg = self.data_receiver.recv()

        return self.uplink_compressor.decompress(uplink_msg), self.downlink_compressor.decompress(downlink_msg)

    def apply_updates_(self, compressed_delta: FloatTensor, compressed_global_delta: FloatTensor):
        # -\gamma g_t^{(i)} is already in grad since compute_thread_
        # add +C[\Delta_t]
        add_grad_dict(self.local_model, grad_dict=self.flattener.unflatten(-compressed_global_delta))
        # add -C[\delta_t^{(i)}]
        add_grad_dict(self.local_model, grad_dict=self.flattener.unflatten(compressed_delta))
        self.local_optimizer.step()

        self.global_optimizer.zero_grad()
        # add +C[\Delta_t]
        add_grad_dict(self.global_model, grad_dict=self.flattener.unflatten(-compressed_global_delta))
        self.global_optimizer.step()


class CocktailGDMaster(Master):
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
        downlink_compressor: Collection[Compressor],
        # Hyperparameters
        gamma: float,
    ):
        self.eval_data = eval_data
        self.global_model = model
        self.global_optimizer = torch.optim.SGD(self.global_model.parameters(), lr=gamma)
        self.flattener = Flattener(shapes={k: v.shape for k, v in self.global_model.named_parameters()})
        self.loss_fn = loss_fn

        self.data_senders = data_senders
        self.data_receivers = data_receivers
        self.uplink_compressors = uplink_compressors
        self.downlink_compressor = downlink_compressor

        self.e = torch.zeros_like(self.flattener.flatten(self.global_model.state_dict()))

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
        self.global_optimizer.zero_grad()
        add_grad_dict(self.global_model, grad_dict=self.flattener.unflatten(-compressed_global_delta))
        self.global_optimizer.step()

        return self.val_loss_()

    def val_loss_(self) -> float:
        with torch.no_grad():
            return float(self.loss_fn(self.global_model(self.eval_data[0]), self.eval_data[1]))


def get_cocktailgd_master_and_clients(
    model: nn.Module,
    loss_fn,
    gamma: float,
    rand_p: float = 0.1,
    top_p: float = 0.2,
    bits: int = 4,
    seed: int = 0,
) -> Tuple[CocktailGDMaster, Collection[CocktailGDClient]]:
    num_clients = len(clients_data)

    uplink_comms = [get_sender_receiver() for _ in range(num_clients)]
    uplink_compressors = [
        CocktailCompressor(rand_p=rand_p, top_p=top_p, bits=bits, seed=seed + i) for i in range(num_clients)
    ]
    downlink_compressor = CocktailCompressor(rand_p=rand_p, top_p=top_p, bits=bits, seed=seed - 1)
    downlink_comms = [get_sender_receiver() for _ in range(num_clients)]
    client_models = [deepcopy(model) for _ in range(num_clients)]

    master = CocktailGDMaster(
        eval_data=eval_data,
        model=model,
        loss_fn=loss_fn,
        data_senders=[s for s, r in downlink_comms],
        data_receivers=[r for s, r in uplink_comms],
        uplink_compressors=uplink_compressors,
        downlink_compressor=downlink_compressor,
        gamma=gamma,
    )

    clients = []
    for i in range(num_clients):
        client = CocktailGDClient(
            data=clients_data[i],
            model=client_models[i],
            loss_fn=loss_fn,
            data_sender=uplink_comms[i][0],
            data_receiver=downlink_comms[i][1],
            uplink_compressor=uplink_compressors[i],
            downlink_compressor=downlink_compressor,
            gamma=gamma,
        )
        clients.append(client)

    return master, clients

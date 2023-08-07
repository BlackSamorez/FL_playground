from copy import deepcopy
from queue import SimpleQueue
from typing import Collection, Mapping, Tuple

import torch
from torch import Tensor, nn

from MiniFL.communications import DataReceiver, DataSender, get_sender_receiver
from MiniFL.compressors import Compressor, IdentityCompressor
from MiniFL.utils import Flattener, add_grad_dict, get_grad_dict


class Client:
    def __init__(
        self,
        lr: float,
        data: Tuple[Tensor, Tensor],
        model: nn.Module,
        loss_fn,
        data_sender: DataSender,
        data_receiver: DataReceiver,
        compressor: Compressor,
    ):
        self.data = data

        self.model = model
        self.flattener = Flattener(shapes={k: v.shape for k, v in self.model.named_parameters()})

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.data_sender = data_sender
        self.data_receiver = data_receiver
        self.compressor = compressor

    def send_grad_get_loss(self):
        self.optimizer.zero_grad()

        loss = self.loss_fn(self.model(self.data[0]), self.data[1])
        loss.backward()

        grad_dict = get_grad_dict(self.model)
        msg = self.compressor.compress(self.flattener.flatten(grad_dict))
        self.data_sender.send(msg)
        return float(loss)

    def apply_global_step(self):
        msg = self.data_receiver.recv()
        grad_dict = self.flattener.unflatten(self.compressor.decompress(msg))

        self.optimizer.zero_grad()
        add_grad_dict(self.model, grad_dict=grad_dict)
        self.optimizer.step()


class Master:
    def __init__(
        self,
        lr: float,
        eval_data: Tuple[Tensor, Tensor],
        model: nn.Module,
        data_senders: Collection[DataSender],
        data_receivers: Collection[DataReceiver],
        compressors: Collection[Compressor],
        loss_fn,
    ):
        self.eval_data = eval_data

        self.model = model
        self.flattener = Flattener(shapes={k: v.shape for k, v in self.model.named_parameters()})

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.data_senders = data_senders
        self.data_receivers = data_receivers
        self.compressors = compressors
        self.loss_fn = loss_fn

    def scale_grads(self, scale: float):
        for v in self.model.parameters():
            v.grad *= scale

    def round(self) -> float:
        self.model.zero_grad()
        for receiver, compressor in zip(self.data_receivers, self.compressors):
            msg = receiver.recv()
            grad_dict = self.flattener.unflatten(compressor.decompress(msg))
            add_grad_dict(self.model, grad_dict=grad_dict)
        self.scale_grads(1 / len(self.data_senders))
        self.optimizer.step()

        for sender, compressor in zip(self.data_senders, self.compressors):
            grad_dict = get_grad_dict(self.model)
            msg = compressor.compress(self.flattener.flatten(grad_dict))
            sender.send(msg)

        with torch.no_grad():
            return float(self.loss_fn(self.model(self.eval_data[0]), self.eval_data[1]))


def get_master_and_clients(
    lr: float,
    clients_data: Collection[Collection[Tuple[Tensor, Tensor]]],
    eval_data: Collection[Tuple[Tensor, Tensor]],
    model: nn.Module,
    loss_fn,
) -> Tuple[Master, Collection[Client]]:
    num_clients = len(clients_data)

    uplink_comms = [get_sender_receiver() for _ in range(num_clients)]
    downlink_comms = [get_sender_receiver() for _ in range(num_clients)]
    compressors = [IdentityCompressor() for _ in range(num_clients)]
    client_models = [deepcopy(model) for _ in range(num_clients)]

    master = Master(
        lr=lr,
        eval_data=eval_data,
        model=model,
        data_senders=[s for s, r in downlink_comms],
        data_receivers=[r for s, r in uplink_comms],
        compressors=compressors,
        loss_fn=loss_fn,
    )

    clients = []
    for i in range(num_clients):
        client = Client(
            lr=lr,
            data=clients_data[i],
            model=client_models[i],
            loss_fn=loss_fn,
            data_sender=uplink_comms[i][0],
            data_receiver=downlink_comms[i][1],
            compressor=compressors[i],
        )
        clients.append(client)

    return master, clients

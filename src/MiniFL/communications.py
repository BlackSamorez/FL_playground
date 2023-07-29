from copy import deepcopy
from queue import SimpleQueue
from typing import Collection, Mapping, Tuple

import torch
from torch import Tensor, nn


def get_num_bits(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits
    else:
        return torch.iinfo(dtype).bits


class DataSender:
    def __init__(self, queue: SimpleQueue) -> None:
        self.queue = queue
        self.n_bits_passed = 0

    def send(self, data: Collection[Tensor]):
        for tensor in data:
            self.n_bits_passed += get_num_bits(tensor.dtype) * tensor.numel()

        self.queue.put(data)


class DataReceiver:
    def __init__(self, queue: SimpleQueue) -> None:
        self.queue = queue

    def recv(self) -> Collection[Tensor]:
        return self.queue.get()


def get_sender_receiver() -> Tuple[DataSender, DataReceiver]:
    queue = SimpleQueue()
    return DataSender(queue=queue), DataReceiver(queue=queue)

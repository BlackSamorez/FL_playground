from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Collection, Mapping, Tuple

import torch
from torch import Tensor, nn
from torch.multiprocessing import Queue as TorchQueue

from .message import Message


class AggregatorSender:
    def __init__(self, queue: TorchQueue, data_example: torch.FloatTensor):
        self.queue = queue
        self.buffer = torch.zeros_like(data_example)
        self.n_bits_passed: float = 0

    def add(self, msg: Message):
        self.n_bits_passed += msg.size
        self.buffer += msg.data

    def flush(self):
        self.queue.put(self.buffer)
        self.buffer = torch.zeros_like(self.buffer)


class AggregatorReciever:
    def __init__(self, queues: Collection[TorchQueue], data_example: torch.FloatTensor):
        self.queues = queues
        self.buffer = torch.zeros_like(data_example)
        self.n_bits_passed: float = 0

    def recieve(self) -> torch.FloatTensor:
        self.buffer.zero_()
        for queue in self.queues:
            self.buffer += queue.get()
        return self.buffer


class BroadcastSender:
    def __init__(self, queues: Collection[TorchQueue]):
        self.queues = queues
        self.n_bits_passed: float = 0

    def broadcast(self, msg: Message):
        self.n_bits_passed += msg.size * len(self.queues)
        msg.data.share_memory_()
        for queue in self.queues:
            queue.put(msg.data)


class BroadcastReceiver:
    def __init__(self, queue: TorchQueue):
        self.queue = queue
        self.n_bits_passed: float = 0

    def recieve(self) -> torch.FloatTensor:
        return self.queue.get()

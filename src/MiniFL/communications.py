from copy import deepcopy
from queue import SimpleQueue
from typing import Collection, Mapping, Tuple

import torch
from torch import Tensor, nn

from .message import Message


class DataSender:
    def __init__(self, queue: SimpleQueue) -> None:
        self.queue = queue
        self.n_bits_passed: float = 0

    def send(self, msg: Message):
        self.n_bits_passed += msg.size
        self.queue.put(msg)


class DataReceiver:
    def __init__(self, queue: SimpleQueue) -> None:
        self.queue = queue
        self.n_bits_passed: float = 0

    def recv(self) -> Message:
        msg = self.queue.get()
        self.n_bits_passed += msg.size
        return msg


def get_sender_receiver() -> Tuple[DataSender, DataReceiver]:
    queue = SimpleQueue()
    return DataSender(queue=queue), DataReceiver(queue=queue)

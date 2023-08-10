from asyncio import Queue
from copy import deepcopy
from typing import Collection, Mapping, Tuple

import torch
from torch import Tensor, nn

from .message import Message


class DataSender:
    def __init__(self, queue: Queue) -> None:
        self.queue = queue
        self.n_bits_passed: float = 0

    async def send(self, msg: Message):
        self.n_bits_passed += msg.size
        await self.queue.put(msg)


class DataReceiver:
    def __init__(self, queue: Queue) -> None:
        self.queue = queue
        self.n_bits_passed: float = 0

    async def recv(self) -> Message:
        msg = await self.queue.get()
        self.n_bits_passed += msg.size
        return msg


def get_sender_receiver() -> Tuple[DataSender, DataReceiver]:
    queue = Queue(maxsize=1)
    return DataSender(queue=queue), DataReceiver(queue=queue)

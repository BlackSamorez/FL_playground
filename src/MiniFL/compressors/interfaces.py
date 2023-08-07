from abc import ABC, abstractmethod
from typing import Collection

from torch import Tensor

from MiniFL.message import Message


class Compressor(ABC):
    @abstractmethod
    def compress(self, data: Collection[Tensor]) -> Message:
        pass

    @abstractmethod
    def decompress(self, msg: Message) -> Collection[Tensor]:
        pass

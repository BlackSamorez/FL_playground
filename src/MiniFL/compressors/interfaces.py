from abc import ABC, abstractmethod
from typing import Collection

from torch import FloatTensor, Tensor

from MiniFL.message import Message


class Compressor(ABC):
    @abstractmethod
    def compress(self, x: FloatTensor) -> Message:
        pass

    @abstractmethod
    def decompress(self, msg: Message) -> FloatTensor:
        pass


class UnbiasedCompressor(Compressor):
    pass

from abc import ABC, abstractmethod
from typing import Collection

from torch import FloatTensor, Tensor

from MiniFL.message import Message


class Compressor(ABC):
    def __init__(self, size: int):
        self.size = size

    @abstractmethod
    def compress(self, x: FloatTensor) -> Message:
        pass


class UnbiasedCompressor(Compressor):
    @abstractmethod
    def omega(self) -> float:
        pass


class ContractiveCompressor(Compressor):
    @abstractmethod
    def alpha(self) -> float:
        pass


class InputVarianceCompressor(Compressor):
    @abstractmethod
    def ab(self) -> (float, float):
        pass

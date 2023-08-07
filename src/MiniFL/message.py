from typing import Collection

from torch import Tensor

FLOAT_SIZE = 32


class Message:
    def __init__(self, data: Collection[Tensor], size: float):
        self.data = data
        self.size = size

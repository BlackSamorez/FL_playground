from torch import FloatTensor

FLOAT_SIZE = 32


class Message:
    def __init__(self, data: FloatTensor, size: float):
        self.data = data
        self.size = size

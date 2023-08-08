from abc import ABC, abstractmethod

from MiniFL.fn import DifferentiableFn
from MiniFL.metrics import ClientStepMetrics, MasterStepMetrics


class Client(ABC):
    def __init__(self, fn: DifferentiableFn):
        self.fn = fn
        self.step_num = 0

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def step(self) -> ClientStepMetrics:
        pass


class Master(ABC):
    def __init__(self, fn: DifferentiableFn):
        self.fn = fn
        self.step_num = 0

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def step(self) -> MasterStepMetrics:
        pass

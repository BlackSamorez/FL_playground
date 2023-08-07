from abc import ABC, abstractmethod


class Client(ABC):
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def step(self) -> float:
        pass


class Master(ABC):
    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def step(self) -> float:
        pass

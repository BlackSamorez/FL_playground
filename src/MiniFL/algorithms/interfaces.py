import threading
from abc import ABC, abstractmethod
from typing import Collection

from tqdm import trange

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


def run_algorithm(master: Master, clients: Collection[Client], num_steps: int):
    def run_client_(steps: int, client: Client):
        client.prepare()
        for _ in range(steps):
            _ = client.step()

    def run_master_(steps: int, master: Master, metrics: list):
        master.prepare()
        for _ in trange(steps):
            master_metrics = master.step()
            metrics.append(master_metrics)

    master_metrics = []

    client_threads = []
    for i, client in enumerate(clients):
        client_threads.append(threading.Thread(target=run_client_, args=(num_steps, client)))
        client_threads[-1].start()

    master_thread = threading.Thread(target=run_master_, args=(num_steps, master, master_metrics))
    master_thread.start()

    master_thread.join()
    for t in client_threads:
        t.join()

    return master_metrics

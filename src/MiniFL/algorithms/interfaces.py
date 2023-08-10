import asyncio
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
    async def prepare(self):
        pass

    @abstractmethod
    async def step(self) -> ClientStepMetrics:
        pass


class Master(ABC):
    def __init__(self, fn: DifferentiableFn):
        self.fn = fn
        self.step_num = 0

    @abstractmethod
    async def prepare(self):
        pass

    @abstractmethod
    async def step(self) -> MasterStepMetrics:
        pass


async def run_algorithm(master: Master, clients: Collection[Client], num_steps: int):
    async def run_client_(steps: int, client: Client):
        await client.prepare()
        for _ in range(steps):
            _ = await client.step()

    async def run_master_(steps: int, master: Master, metrics: list):
        await master.prepare()
        for _ in trange(steps):
            master_metrics = await master.step()
            metrics.append(master_metrics)

    master_metrics = []

    loop = asyncio.get_event_loop()

    tasks = []
    for i, client in enumerate(clients):
        tasks.append(loop.create_task(run_client_(num_steps, client)))
    tasks.append(loop.create_task(run_master_(num_steps, master, master_metrics)))
    await asyncio.wait(tasks)

    return master_metrics

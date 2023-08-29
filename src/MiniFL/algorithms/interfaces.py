import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.pool import Pool, ThreadPool
from queue import SimpleQueue
from typing import Collection

import torch
from torch import FloatTensor
from tqdm import trange

from MiniFL.communications import AggregatorReciever, AggregatorSender, BroadcastReceiver, BroadcastSender
from MiniFL.fn import DifferentiableFn
from MiniFL.message import Message
from MiniFL.metrics import ClientStepMetrics, MasterStepMetrics


class Client(ABC):
    def __init__(self, fn: DifferentiableFn):
        self.fn = fn
        self.step_num = 0

    @abstractmethod
    def step(self, broadcasted_master_tensor: FloatTensor) -> (Message, FloatTensor, ClientStepMetrics):
        pass


class Master(ABC):
    def __init__(self, size: int, num_clients: int):
        self.size = size
        self.num_clients = num_clients
        self.step_num = 0

    @abstractmethod
    def step(self, sum_worker_tensor: FloatTensor) -> Message:
        pass


def worker_process_(client, broadcasted_master_tensor):
    return client.step(broadcasted_master_tensor)


def run_algorithm_sequantially(master: Master, clients: Collection[Client], num_steps: int):
    total_bits_uplink = 0
    total_bits_downlink = 0
    master_metrics = []

    broadcasted_master_tensor = torch.zeros(master.size)
    sum_worker_tensors = torch.zeros(master.size)
    for step in trange(num_steps):
        worker_results = [worker_process_(client, broadcasted_master_tensor) for client in clients]
        total_grad_norm = torch.linalg.vector_norm(sum(result[1] for result in worker_results)) / len(clients)
        total_value = sum(result[2].value for result in worker_results) / len(clients)

        sum_worker_tensors = sum(result[0].data for result in worker_results)
        total_bits_uplink += sum(result[0].size for result in worker_results)

        broadcasted_master_msg = master.step(sum_worker_tensors)
        broadcasted_master_tensor = broadcasted_master_msg.data
        total_bits_downlink += broadcasted_master_msg.size

        master_metrics.append(
            MasterStepMetrics(
                step=step,
                value=total_value,
                grad_norm=total_grad_norm,
                total_bits_received=total_bits_uplink,
                total_bits_sent=total_bits_downlink,
            )
        )

    return master_metrics


def run_algorithm_with_threads(master: Master, clients: Collection[Client], num_steps: int, num_threads: int = 10):
    total_bits_uplink = 0
    total_bits_downlink = 0
    master_metrics = []
    with ThreadPool(num_threads) as pool:
        broadcasted_master_tensor = torch.zeros(master.size)
        broadcasted_master_tensor.share_memory_()
        sum_worker_tensors = torch.zeros(master.size)
        for step in trange(num_steps):
            worker_results = pool.starmap(
                worker_process_,
                [(client, broadcasted_master_tensor) for client in clients],
            )
            total_grad_norm = torch.linalg.vector_norm(sum(result[1] for result in worker_results)) / len(clients)
            total_value = sum(result[2].value for result in worker_results) / len(clients)

            sum_worker_tensors = sum(result[0].data for result in worker_results)
            total_bits_uplink += sum(result[0].size for result in worker_results)

            broadcasted_master_msg = master.step(sum_worker_tensors)
            broadcasted_master_tensor = broadcasted_master_msg.data
            total_bits_downlink += broadcasted_master_msg.size

            master_metrics.append(
                MasterStepMetrics(
                    step=step,
                    value=total_value,
                    grad_norm=total_grad_norm,
                    total_bits_received=total_bits_uplink,
                    total_bits_sent=total_bits_downlink,
                )
            )

    return master_metrics


def run_algorithm_with_processes(master: Master, clients: Collection[Client], num_steps: int, num_processes: int = 10):
    total_bits_uplink = 0
    total_bits_downlink = 0
    master_metrics = []
    with Pool(num_processes) as pool:
        broadcasted_master_tensor = torch.zeros(master.size)
        broadcasted_master_tensor.share_memory_()
        sum_worker_tensors = torch.zeros(master.size)
        for step in trange(num_steps):
            worker_results = pool.starmap(
                worker_process_,
                [(client, broadcasted_master_tensor) for client in clients],
            )
            total_grad_norm = torch.linalg.vector_norm(sum(result[1] for result in worker_results)) / len(clients)
            total_value = sum(result[2].value for result in worker_results) / len(clients)

            sum_worker_tensors = sum(result[0].data for result in worker_results)
            total_bits_uplink += sum(result[0].size for result in worker_results)

            broadcasted_master_msg = master.step(sum_worker_tensors)
            broadcasted_master_tensor = broadcasted_master_msg.data
            total_bits_downlink += broadcasted_master_msg.size

            master_metrics.append(
                MasterStepMetrics(
                    step=step,
                    value=total_value,
                    grad_norm=total_grad_norm,
                    total_bits_received=total_bits_uplink,
                    total_bits_sent=total_bits_downlink,
                )
            )

    return master_metrics

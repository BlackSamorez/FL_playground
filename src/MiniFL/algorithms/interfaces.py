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
    def step(self, broadcasted_master_tensor: FloatTensor) -> (Message, ClientStepMetrics):
        pass


class Master(ABC):
    def __init__(self, fn: DifferentiableFn, num_clients: int):
        self.fn = fn
        self.num_clients = num_clients
        self.step_num = 0

    @abstractmethod
    def step(self, sum_worker_tensor: FloatTensor) -> (Message, MasterStepMetrics):
        pass


def worker_process_(client, broadcasted_master_tensor):
    return client.step(broadcasted_master_tensor)


def run_algorithm_sequantially(master: Master, clients: Collection[Client], num_steps: int):
    total_bits_uplink = 0
    total_bits_downlink = 0
    master_metrics = []

    broadcasted_master_tensor = torch.zeros_like(master.fn.zero_like_grad())
    sum_worker_tensors = torch.zeros_like(master.fn.zero_like_grad())
    for step in trange(num_steps):
        worker_results = [worker_process_(client, broadcasted_master_tensor) for client in clients]
        sum_worker_tensors = sum(result[0].data for result in worker_results)
        total_bits_uplink += sum(result[0].size for result in worker_results)

        master_result = master.step(sum_worker_tensors)
        broadcasted_master_tensor = master_result[0].data
        total_bits_downlink += master_result[0].size
        master_metrics_ = master_result[1]
        master_metrics_.total_bits_received = total_bits_uplink
        master_metrics_.total_bits_sent = total_bits_downlink
        master_metrics.append(master_metrics_)

    return master_metrics


def run_algorithm_with_threads(master: Master, clients: Collection[Client], num_steps: int, num_threads: int = 10):
    total_bits_uplink = 0
    total_bits_downlink = 0
    master_metrics = []
    with ThreadPool(num_threads) as pool:
        broadcasted_master_tensor = torch.zeros_like(master.fn.zero_like_grad())
        broadcasted_master_tensor.share_memory_()
        sum_worker_tensors = torch.zeros_like(master.fn.zero_like_grad())
        for step in trange(num_steps):
            worker_results = pool.starmap(
                worker_process_,
                [(client, broadcasted_master_tensor) for client in clients],
            )
            sum_worker_tensors = sum(result[0].data for result in worker_results)
            total_bits_uplink += sum(result[0].size for result in worker_results)

            master_result = master.step(sum_worker_tensors)
            broadcasted_master_tensor = master_result[0].data
            total_bits_downlink += master_result[0].size
            master_metrics_ = master_result[1]
            master_metrics_.total_bits_received = total_bits_uplink
            master_metrics_.total_bits_sent = total_bits_downlink
            master_metrics.append(master_metrics_)

    return master_metrics


def run_algorithm_with_processes(master: Master, clients: Collection[Client], num_steps: int, num_processes: int = 10):
    total_bits_uplink = 0
    total_bits_downlink = 0
    master_metrics = []
    with Pool(num_processes) as pool:
        broadcasted_master_tensor = torch.zeros_like(master.fn.zero_like_grad())
        broadcasted_master_tensor.share_memory_()
        sum_worker_tensors = torch.zeros_like(master.fn.zero_like_grad())
        for step in trange(num_steps):
            worker_results = pool.starmap(
                worker_process_,
                [(client, broadcasted_master_tensor) for client in clients],
            )
            sum_worker_tensors = sum(result[0].data for result in worker_results)
            total_bits_uplink += sum(result[0].size for result in worker_results)

            master_result = master.step(sum_worker_tensors)
            broadcasted_master_tensor = master_result[0].data
            total_bits_downlink += master_result[0].size
            master_metrics_ = master_result[1]
            master_metrics_.total_bits_received = total_bits_uplink
            master_metrics_.total_bits_sent = total_bits_downlink
            master_metrics.append(master_metrics_)

    return master_metrics

import multiprocessing
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import SimpleQueue
from typing import Collection

import torch
from torch import FloatTensor
from torch.multiprocessing import Queue as TorchQueue
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


def run_algorithm_with_threads(master: Master, clients: Collection[Client], num_steps: int):
    num_clients = len(clients)
    uplink_comms = [SimpleQueue() for _ in range(num_clients)]
    downlink_comms = [SimpleQueue() for _ in range(num_clients)]

    def run_master_(steps: int, master: Master, metrics: list):
        sum_worker_tensor = torch.empty_like(master.fn.zero_like_grad())
        total_bits_received = 0
        for _ in trange(steps):
            sum_worker_tensor.zero_()
            for comm in uplink_comms:
                msg = comm.get()
                sum_worker_tensor += msg.data
                total_bits_received += msg.size

            msg, master_metrics = master.step(sum_worker_tensor)
            master_metrics.total_bits_received = total_bits_received
            metrics.append(master_metrics)
            for comm in downlink_comms:
                comm.put(msg.data)

    def run_client_(i: int, steps: int, client: Client):
        broadcasted_master_tensor = torch.zeros_like(master.fn.zero_like_grad())
        for _ in range(steps):
            msg, client_metrics = client.step(broadcasted_master_tensor)
            uplink_comms[i].put(msg)
            msg = downlink_comms[i].get()
            broadcasted_master_tensor = msg.data

    master_metrics = []

    client_threads = []
    for i, client in enumerate(clients):
        client_threads.append(threading.Thread(target=run_client_, args=(i, num_steps, client)))
        client_threads[-1].start()

    master_thread = threading.Thread(target=run_master_, args=(num_steps, master, master_metrics))
    master_thread.start()

    master_thread.join()
    for t in client_threads:
        t.join()

    return master_metrics


def worker_process_(uplink_queue, downlink_queue, clients: Collection[Client], num_steps: int):
    aggregator_sender = AggregatorSender(uplink_queue, clients[0].fn.zero_like_grad())
    broadcast_receiver = BroadcastReceiver(downlink_queue)
    broadcasted_master_tensor = torch.zeros_like(clients[0].fn.zero_like_grad())
    for _ in range(num_steps):
        for client in clients:
            msg, client_metrics = client.step(broadcasted_master_tensor)
            aggregator_sender.add(msg)
        aggregator_sender.flush()
        broadcasted_master_tensor = broadcast_receiver.recieve()


def master_process_(uplink_queues, downlink_queues, master: Master, num_steps: int):
    metrics = []
    aggregator_receiver = AggregatorReciever(uplink_queues, master.fn.zero_like_grad())
    broadcast_sender = BroadcastSender(downlink_queues)
    sum_worker_tensor = torch.empty_like(master.fn.zero_like_grad())
    for _ in trange(num_steps):
        sum_worker_tensor = aggregator_receiver.recieve()
        msg, master_metrics = master.step(sum_worker_tensor)
        master_metrics.total_bits_received = aggregator_receiver.n_bits_passed
        metrics.append(master_metrics)
        broadcast_sender.broadcast(msg)
    return metrics


def run_algorithm_with_processes(master: Master, clients: Collection[Client], num_steps: int, num_processes: int = 10):
    with multiprocessing.Manager() as manager:
        uplink_queues = [manager.Queue(1) for _ in range(num_processes)]
        downlink_queues = [manager.Queue(1) for _ in range(num_processes)]

        with multiprocessing.Pool(num_processes) as pool:
            result = pool.starmap_async(
                worker_process_,
                [
                    (uplink_queues[i], downlink_queues[i], clients[i::num_processes], num_steps)
                    for i in range(num_processes)
                ],
            )

            master_process_(uplink_queues, downlink_queues, master, num_steps)

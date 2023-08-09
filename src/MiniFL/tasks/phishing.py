import os
from copy import deepcopy
from typing import Collection, Tuple

import torch
from sklearn.datasets import load_svmlight_file

from MiniFL.fn import DifferentiableFn, NNDifferentiableFn

PHISHING_NUM_FEATURES = 68
PHISHING_NUM_DATAPOINTS = 11055


def get_phishing_fns(
    data_path: os.PathLike, model: torch.nn.Module, num_clients: int, batch_size: int, seed: int = 0
) -> Tuple[DifferentiableFn, Collection[DifferentiableFn]]:
    data, labels = load_svmlight_file(data_path)
    enc_labels = labels.copy()
    data_dense = data.todense()

    eval_data = (
        torch.from_numpy(data_dense).to(torch.float32),
        torch.from_numpy(enc_labels).to(torch.float32)[:, None],
    )
    clients_data = [
        (x, y)
        for x, y in zip(
            torch.tensor_split(eval_data[0], num_clients, dim=0), torch.tensor_split(eval_data[1], num_clients, dim=0)
        )
    ]

    loss_fn = torch.nn.BCEWithLogitsLoss()

    master_fn = NNDifferentiableFn(
        model=deepcopy(model),
        data=eval_data,
        loss_fn=loss_fn,
        batch_size=batch_size,
        seed=seed,
    )

    client_fns = [
        NNDifferentiableFn(
            model=deepcopy(model),
            data=clients_data[i],
            loss_fn=loss_fn,
            batch_size=batch_size,
            seed=seed + i + 1,
        )
        for i in range(num_clients)
    ]
    return master_fn, client_fns


def get_phishing_regression_fns(
    data_path: os.PathLike, num_clients: int, batch_size: int, seed: int = 0
) -> Tuple[DifferentiableFn, Collection[DifferentiableFn]]:
    return get_phishing_fns(
        data_path=data_path,
        model=torch.nn.Linear(PHISHING_NUM_FEATURES, 1, bias=False),
        num_clients=num_clients,
        batch_size=batch_size,
        seed=seed,
    )

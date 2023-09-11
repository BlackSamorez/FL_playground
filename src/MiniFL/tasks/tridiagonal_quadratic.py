import math
from typing import Collection

import numpy as np
import scipy
import torch
from torch import FloatTensor

from MiniFL.fn import AutogradDifferentiableFn, DifferentiableFn, MeanDifferentiableFn, ReplicatedFn


class TridiagonalQuadraticFn(DifferentiableFn):
    def __init__(self, main_diag, side_diag, b):
        self.main_diag = main_diag
        self.side_diag = side_diag
        self.A = scipy.sparse.diags([side_diag, main_diag, side_diag], [-1, 0, 1])
        self.b = b

        self.point = np.zeros_like(self.b)

    def is_full_grad(self) -> bool:
        return True

    def get_value(self) -> float:
        return (1 / 2.0) * np.dot(self.point, self.A.dot(self.point)) - np.dot(self.b, self.point)

    def get_parameters(self) -> FloatTensor:
        return torch.tensor(self.point)

    def get_flat_grad_estimate(self) -> FloatTensor:
        return torch.tensor(self.A.dot(self.point) - self.b)

    def step(self, delta: FloatTensor):
        self.point += delta.numpy()

    def zero_like_grad(self) -> FloatTensor:
        return torch.zeros(self.point.shape)

    def size(self) -> int:
        return len(self.point)

    def liptschitz_gradient_constant(self) -> float:
        eigvals = scipy.linalg.eigh_tridiagonal(self.main_diag, self.side_diag, eigvals_only=True)
        return max(abs(eigvals))

    @staticmethod
    def smoothness_variance(fns: Collection[DifferentiableFn]) -> float:
        weights = np.ones(len(fns), dtype=np.float32) / len(fns)
        for fn in fns:
            assert isinstance(fn, TridiagonalQuadraticFn)
        matrices = [fn.A for fn in fns]

        matrix_square_ = lambda A: A @ A

        def matrix_mean_(A_list, weights=None):
            if weights is not None:
                A_list_weights = [A / (len(A_list) * weight) for A, weight in zip(A_list, weights)]
            else:
                A_list_weights = A_list
            return sum(A_list_weights) / len(A_list)

        mean_matrix = matrix_mean_(matrices)
        square_mean_matrix = matrix_square_(mean_matrix)
        square_matrices = list(map(matrix_square_, matrices))
        mean_square_matrix = matrix_mean_(square_matrices, weights)
        result_matrix = mean_square_matrix - square_mean_matrix

        svdvals = scipy.sparse.linalg.svds(result_matrix, k=1, return_singular_vectors=False)
        op_norm = max(svdvals)
        svb = op_norm ** (1 / 2)
        return svb


def create_worst_case(dim, liptschitz_gradient_constant, noise_lambda=0, seed=None, strategy="mul"):
    scale = liptschitz_gradient_constant / 4.0
    main_diag = 2 * np.ones(dim, dtype=np.float32)
    side_diag = -1 * np.ones(dim - 1, dtype=np.float32)
    b = np.zeros(dim, dtype=np.float32)
    b[0] = -1
    if noise_lambda > 0:
        generator = np.random.default_rng(seed=seed)
        if strategy == "add":
            noise = noise_lambda * generator.standard_exponential(size=(1,), dtype=np.float32)
            main_diag += noise
            side_diag += noise
        if strategy == "mul":
            noise_scale = 1 + noise_lambda * generator.standard_normal(size=(1,), dtype=np.float32)
            noise_bias = noise_lambda * generator.standard_normal(size=(1,), dtype=np.float32)
            b[0] += noise_bias
            b[0] *= noise_scale
            main_diag *= noise_scale
            side_diag *= noise_scale
    b[0] *= scale
    return scale * main_diag, scale * side_diag, b


def create_worst_case_tridiagonal_quadratics(
    num_clients: int,
    size: int,
    liptschitz_gradient_constant: float = 1,
    noise_lambda: float = 0.1,
    strongly_convex_constant: float = 1e-6,
    seed: int = 0,
):
    if noise_lambda == 0:
        main_diag, side_diag, b = create_worst_case(size, liptschitz_gradient_constant, noise_lambda, seed)
        min_eig = min(scipy.linalg.eigh_tridiagonal(main_diag, side_diag, eigvals_only=True))
        main_diag = main_diag - min_eig
        main_diag = main_diag + strongly_convex_constant

        return [
            ReplicatedFn(
                TridiagonalQuadraticFn(
                    main_diag=main_diag,
                    side_diag=side_diag,
                    b=b,
                ),
                world_size=num_clients,
            )
        ] * num_clients

    main_diags = []
    side_diags = []
    bs = []
    for i in range(num_clients):
        main_diag, side_diag, b = create_worst_case(size, liptschitz_gradient_constant, noise_lambda, seed + i)
        main_diags.append(main_diag)
        side_diags.append(side_diag)
        bs.append(b)
    lambda_mean_matrices = lambda A_list: sum(A_list) / len(A_list)
    mean_main_diag = lambda_mean_matrices(main_diags)
    mean_side_diag = lambda_mean_matrices(side_diags)
    eigs = scipy.linalg.eigh_tridiagonal(mean_main_diag, mean_side_diag, eigvals_only=True)
    min_eig = min(eigs)
    funcs = []
    for main_diag, side_diag, b in zip(main_diags, side_diags, bs):
        main_diag = main_diag - min_eig
        main_diag = main_diag + strongly_convex_constant
        funcs.append(TridiagonalQuadraticFn(main_diag, side_diag, b))
    return funcs

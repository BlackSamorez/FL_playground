from dataclasses import dataclass


@dataclass
class ClientStepMetrics:
    step: int
    value: float
    total_bits_sent: float
    total_bits_received: float
    grad_norm: float


@dataclass
class MasterStepMetrics:
    step: int
    value: float
    total_bits_sent: float
    total_bits_received: float
    grad_norm: float

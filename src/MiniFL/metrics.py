from dataclasses import dataclass


@dataclass
class ClientStepMetrics:
    step: int
    value: float
    grad_norm: float
    total_bits_sent: float = 0
    total_bits_received: float = 0


@dataclass
class MasterStepMetrics:
    step: int
    value: float
    grad_norm: float
    total_bits_sent: float = 0
    total_bits_received: float = 0

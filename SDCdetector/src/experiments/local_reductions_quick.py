"""Small and quick workload for local reductions experiments"""
from dataclasses import dataclass, field

import torch

from experiments.local_reductions import LocalReductions


@dataclass
class LocalReductionsQuick(LocalReductions):
    rows: int = 16
    widths: tuple[int, ...] = (1024,)
    iterations: int = 1
    eps: float = 1e-05
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32])


EXPERIMENT = LocalReductionsQuick()

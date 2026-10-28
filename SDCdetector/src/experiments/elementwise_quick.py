"""Small and quick workload for elementwise experiments"""
from dataclasses import dataclass, field

import torch

from experiments.elementwise import Elementwise


@dataclass
class ElementwiseQuick(Elementwise):
    numel: int = 64 * 1024
    iterations: int = 1
    chain_steps: int = 1
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32, torch.bfloat16])
    integer_dtypes: tuple[torch.dtype, ...] = (torch.int64,)


EXPERIMENT = ElementwiseQuick()

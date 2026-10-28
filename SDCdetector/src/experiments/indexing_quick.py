"""Small and quick workload for indexing experiments"""
from dataclasses import dataclass, field

import torch

from experiments.indexing import Indexing


@dataclass
class IndexingQuick(Indexing):
    shapes: tuple[tuple[int, int], ...] = ((127, 257),)
    iterations: int = 1
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.int64])


EXPERIMENT = IndexingQuick()

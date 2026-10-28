"""Small and quick workload for fused attention experiments"""
from dataclasses import dataclass, field

import torch

from experiments.fused_attention import FusedAttention


@dataclass
class FusedAttentionQuick(FusedAttention):
    batch_size: int = 2
    heads: int = 4
    lengths: tuple[int, ...] = (128,)
    head_dims: tuple[int, ...] = (64,)
    causal_modes: tuple[bool, ...] = (False, True)
    iterations: int = 1
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.bfloat16])


EXPERIMENT = FusedAttentionQuick()

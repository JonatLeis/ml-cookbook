"""Dao-AILab FlashAttention 3 replica screening."""
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module

from experiments._flash_attention import FlashAttention


@lru_cache(maxsize=1)
def _load_backend():
    return import_module("flash_attn_3.flash_attn_interface").flash_attn_func


@dataclass
class FA3(FlashAttention):
    name = "fa3"
    backend = "flash_attn_3.flash_attn_interface"
    # The supplied FA3 build rejects deterministic backward at head_dim=256.
    shapes: tuple[tuple[int, int, int, int], ...] = FlashAttention.shapes[:-1]

    def _load_backend(self):
        return _load_backend()


EXPERIMENT = FA3()

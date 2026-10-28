"""Dao-AILab FlashAttention 4 screening with an optional SM90 backward workaround."""
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module

from experiments._flash_attention import FlashAttention
from experiments.fa4_compat import stat_loads
from reporting import diagnostic


@lru_cache(maxsize=1)
def _load_backend():
    return import_module("flash_attn.cute").flash_attn_func


@dataclass
class FA4(FlashAttention):
    name = "fa4"
    backend = "flash_attn.cute"
    workaround: bool = True

    def _load_backend(self):
        return _load_backend()

    def _check_config(self):
        super()._check_config()
        if type(self.workaround) is not bool:
            raise ValueError("workaround must be boolean")

    def _backward_context(self, device):
        context = stat_loads(device) if self.workaround else None
        diagnostic("FA4_STAT_LOADS", workaround=self.workaround,
                   mode="direct" if context is not None else "original")
        return context


EXPERIMENT = FA4()

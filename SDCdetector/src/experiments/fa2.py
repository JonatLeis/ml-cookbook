"""Dao-AILab FlashAttention 2 replica screening."""
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.metadata import version as package_version

from experiments._flash_attention import FlashAttention


@lru_cache(maxsize=1)
def _load_backend():
    installed = package_version("flash-attn")
    if installed.split(".")[0] != "2":
        raise RuntimeError(f"FA2 requested, but flash-attn {installed} is installed")
    return import_module("flash_attn.flash_attn_interface").flash_attn_func


@dataclass
class FA2(FlashAttention):
    name = "fa2"
    backend = "flash_attn.flash_attn_interface"

    def _load_backend(self):
        return _load_backend()

    def _options(self, causal):
        return {**super()._options(causal), "dropout_p": 0.0}


EXPERIMENT = FA2()

"""Scoped SM90 statistic loads for FA4 backward; no CUDA imports until selected. -- Fix for FA4 backward on Hopper GPUs"""
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module
import threading

import torch

from reporting import diagnostic


def _make_stat_copy(stats):
    # Keep this instruction sequence identical to src_test/fa_test.py's tested
    # direct-stat-load probe. The original transaction barrier/count is retained.
    import cutlass.cute as cute
    from cutlass import Float32, Int32
    from cutlass.cutlass_dsl import dsl_user_op
    from cutlass._mlir.dialects import llvm

    @dsl_user_op
    def load(gmem_ptr: cute.Pointer, smem_ptr: cute.Pointer,
             tma_bar_ptr: cute.Pointer, size: int, *, loc=None, ip=None):
        if not isinstance(size, int) or size <= 0 or size % 16:
            raise ValueError("direct stat loads require a static positive multiple of 16 bytes")
        stats["compile_calls"] += 1
        llvm.inline_asm(
            None,
            [gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
             smem_ptr.toint(loc=loc, ip=ip).ir_value(),
             tma_bar_ptr.toint(loc=loc, ip=ip).ir_value(),
             Int32(size).ir_value(loc=loc, ip=ip)],
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
            "[$1], [$0], $3, [$2];",
            "l,r,r,r,~{memory}", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    def factory(src_tensor, dst_tensor, single_stage=False, **kwargs):
        if (single_stage or kwargs or src_tensor.element_type != Float32
                or dst_tensor.element_type != Float32
                or src_tensor.memspace != cute.AddressSpace.gmem
                or dst_tensor.memspace != cute.AddressSpace.smem):
            raise ValueError("direct stat loads only support staged FP32 GMEM-to-SMEM loads")
        src = cute.group_modes(src_tensor, 0, cute.rank(src_tensor) - 1)
        dst = cute.group_modes(dst_tensor, 0, cute.rank(dst_tensor) - 1)
        size = cute.size(src.shape[:-1]) * Float32.width // 8
        if cute.size(dst.shape[:-1]) * Float32.width // 8 != size:
            raise ValueError("direct stat loads require equal source/destination tile sizes")

        def copy_bulk(src_idx, dst_idx, tma_bar_ptr):
            with cute.arch.elect_one():
                load(src[None, src_idx].iterator, dst[None, dst_idx].iterator,
                     tma_bar_ptr, size)

        return copy_bulk

    return factory


class _SM90StatLoads:
    def __init__(self):
        # Use the backend's actual dependency (quack.copy_utils in the tested
        # build), rather than the similarly named flash_attn.cute.copy_utils.
        self.owner = import_module("flash_attn.cute.flash_bwd_sm90").copy_utils
        self.backward = import_module("flash_attn.cute.interface")._flash_attn_bwd
        if not callable(self.owner.cpasync_bulk_get_copy_fn) or not hasattr(self.backward, "compile_cache"):
            raise RuntimeError("installed FA4 does not expose the expected SM90 backward helpers/cache")
        self.stats = {"compile_calls": 0}
        self.copy = _make_stat_copy(self.stats)
        # FA4 keys do not include our helper replacement. Swap in a private
        # mapping, never clear/export the library's potentially persistent cache.
        self.cache = {}
        self.lock = threading.RLock()

    @contextmanager
    def applied(self):
        with self.lock:
            original_copy = self.owner.cpasync_bulk_get_copy_fn
            original_cache = self.backward.compile_cache
            previous_keys = set(self.cache)
            previous_calls = self.stats["compile_calls"]
            try:
                self.owner.cpasync_bulk_get_copy_fn = self.copy
                self.backward.compile_cache = self.cache
                yield
                if (not self.stats["compile_calls"] or
                        (self.cache.keys() - previous_keys and self.stats["compile_calls"] == previous_calls)):
                    raise RuntimeError("FA4 direct stat loads were not compiled for the backward kernel")
                diagnostic("FA4_STAT_LOADS", mode="direct", compile_calls=self.stats["compile_calls"],
                           cached_kernels=len(self.cache))
            except BaseException:
                # A failed/unsupported compilation must not be reused later.
                self.cache.clear()
                self.stats["compile_calls"] = 0
                raise
            finally:
                self.backward.compile_cache = original_cache
                self.owner.cpasync_bulk_get_copy_fn = original_copy


@lru_cache(maxsize=1)
def _sm90_stat_loads():
    # Preserve patched JIT kernels across cases and whole-experiment repetitions.
    return _SM90StatLoads()


def stat_loads(device):
    """Return the direct-load context; callers explicitly opt in on SM90."""
    sm90 = device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)
    if not sm90:
        raise ValueError("FA4 workaround=true requires an SM90 GPU (H100/H200); "
                         "set workaround=false on other architectures (e.g. Blackwell)")
    return _sm90_stat_loads().applied

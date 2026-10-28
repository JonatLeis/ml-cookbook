from dataclasses import dataclass
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from prometheus_client.parser import text_string_to_metric_families

from experiment import Experiment
from validation import ValidationError, assert_exact
from experiments._utils import NCCL_ALLREDUCE_DTYPES, gather_rank_tensors, log_once


@dataclass
class AllReduceTest(Experiment):
    """Validate AllReduce SUM separately for every configured dtype."""

    numel: int = 16 * 1024 * 1024
    warmup: int = 5
    iterations: int = 20

    @staticmethod
    def _check_expected_sum(dtype, expected_sum):
        """Require exact representation of all positive integer partial sums."""
        if dtype == torch.bool or dtype.is_complex:
            raise ValidationError(f"{dtype}: this integer-sum test requires a real numeric dtype")
        if dtype.is_floating_point:
            info = torch.finfo(dtype)
            limit = min(info.max, int(2 / info.eps))
        else:
            limit = torch.iinfo(dtype).max
        if expected_sum > limit:
            raise ValidationError(
                f"{dtype}: expected sum {expected_sum} exceeds the safe exact "
                f"integer range of this test (maximum {limit})"
            )

    def run(self, device, seed) -> dict[str, torch.Tensor]:
        if not dist.is_initialized():
            raise ValidationError("This experiment requires torchrun and an initialized process group")

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if world_size < 2:
            raise ValidationError("This experiment requires at least two ranks")
        if self.numel < 1 or self.warmup < 0 or self.iterations < 1:
            raise ValidationError("Invalid tensor size or iteration count")

        dtypes = NCCL_ALLREDUCE_DTYPES
        if not dtypes or len(set(dtypes)) != len(dtypes):
            raise ValidationError("Configured AllReduce dtypes must be nonempty and unique")

        expected_sum = world_size * (world_size + 1) // 2
        # Check every dtype before starting any collective for this test.
        for dtype in dtypes:
            self._check_expected_sum(dtype, expected_sum)

        results: dict[str, torch.Tensor] = {}
        for dtype in dtypes:
            result = torch.empty(self.numel, dtype=dtype, device=device)
            if not result.is_cuda:
                raise ValidationError("This profiling experiment requires a CUDA device")
            torch.cuda.set_device(result.device)

            with torch.cuda.nvtx.range(f"allreduce/{dtype}/warmup"):
                for _ in range(self.warmup):
                    result.fill_(rank + 1)
                    dist.all_reduce(result, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize(result.device)
            dist.barrier(device_ids=[result.device.index])

            with torch.cuda.nvtx.range(f"allreduce/{dtype}"):
                for step in range(self.iterations):
                    result.fill_(rank + 1)
                    with torch.cuda.nvtx.range(f"allreduce/iteration_{step:03d}"):
                        dist.all_reduce(result, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize(result.device)

            results.update({f"{dtype}/{name}": tensor
                            for name, tensor in gather_rank_tensors(result).items()})

        time.sleep(35)
        # This remains one aggregate flag, not evidence for each dtype/rank.
        observed_algorithms = set()
        prometheus_dir = os.getenv("NCCL_INSPECTOR_DUMP_DIR")
        if not prometheus_dir:
            log_once("ALLREDUCE --- NVLS evidence unavailable: dump directory is not configured")
        else:
            try:
                for path in sorted(Path(prometheus_dir).glob("*.prom")):
                    try:
                        file_algorithms = set()
                        for family in text_string_to_metric_families(path.read_text(encoding="utf-8")):
                            for sample in family.samples:
                                labels = sample.labels
                                if labels.get("collective", "").lower() == "allreduce" and sample.value > 0:
                                    file_algorithms.add(labels.get("algo_proto", "").lower())
                        observed_algorithms.update(file_algorithms)
                    except (OSError, ValueError) as exc:
                        observed_algorithms.add("")
                        log_once(f"ALLREDUCE --- Cannot evaluate {path.name}: {exc}")
            except OSError as exc:
                log_once(f"ALLREDUCE --- Cannot read Inspector directory: {exc}")

        nvls_used = bool(observed_algorithms) and all(
            "nvls" in algorithm for algorithm in observed_algorithms
        )
        torch.cuda.synchronize(result.device)
        dist.barrier(device_ids=[result.device.index])
        results["nvls_used"] = torch.tensor(nvls_used, dtype=torch.bool)
        return results

    def validate(self, results: dict[str, torch.Tensor]):
        if not isinstance(results, dict):
            raise ValidationError("AllReduce results must be a dictionary")
        for name, tensor in results.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValidationError(
                    f"{name!r}: results must be a flat dict[str, torch.Tensor]"
                )

        world_size = dist.get_world_size()
        expected_sum = world_size * (world_size + 1) // 2
        expected_ranks = {f"rank_{index:04d}" for index in range(world_size)}
        expected_dtypes = tuple(
            str(dtype) for dtype in NCCL_ALLREDUCE_DTYPES
        )
        expected_keys = {
            f"{dtype_name}/{rank_name}"
            for dtype_name in expected_dtypes
            for rank_name in expected_ranks
        } | {"nvls_used"}
        if not expected_dtypes or set(results) != expected_keys:
            raise ValidationError(
                f"Incorrect result keys: missing={sorted(expected_keys - set(results))}, "
                f"unexpected={sorted(set(results) - expected_keys)}"
            )

        for dtype_name in expected_dtypes:
            reference = results[f"{dtype_name}/rank_0000"]
            if str(reference.dtype) != dtype_name:
                raise ValidationError(
                    f"{dtype_name}/rank_0000: unexpected tensor dtype {reference.dtype}"
                )
            self._check_expected_sum(reference.dtype, expected_sum)
            expected = torch.full_like(reference, expected_sum)

            for rank_name in sorted(expected_ranks):
                context = f"{dtype_name}/{rank_name}"
                result = results[context]
                if result.dtype != reference.dtype:
                    raise ValidationError(f"{context}: unexpected tensor dtype {result.dtype}")
                if tuple(result.shape) != (self.numel,):
                    raise ValidationError(
                        f"{context}: expected shape {(self.numel,)}, got {tuple(result.shape)}"
                    )
                try:
                    assert_exact({"allreduce": expected}, {"allreduce": result})
                except ValidationError as exc:
                    raise ValidationError(f"{context}: incorrect AllReduce result: {exc}") from exc

            log_once(f"ALLREDUCE --- {dtype_name}: all ranks match the expected sum exactly")

        # Metadata is a tensor too, but has its own validation rules.
        nvls_used = results.get("nvls_used")
        if (
            not isinstance(nvls_used, torch.Tensor)
            or nvls_used.dtype != torch.bool
            or nvls_used.ndim != 0
        ):
            raise ValidationError("nvls_used: expected a scalar boolean tensor")
        if not nvls_used.item():
            raise ValidationError("nvls_used: NVLS usage could not be confirmed for all observed AllReduce samples")

        # Preserve the flat schema expected by the surrounding framework.
        super().validate(results)


EXPERIMENT = AllReduceTest()

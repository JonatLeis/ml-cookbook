"""Tests indexing, gather, scatter, and strided reads/writes with various dtypes and shapes."""
from dataclasses import dataclass, field

import torch

from reporting import progress
from experiments._utils import SDCChecks, ScreeningExperiment, check_dtypes, cpu_generator


@dataclass
class Indexing(ScreeningExperiment):
    shapes: tuple[tuple[int, int], ...] = ((257, 511), (1024, 1024))
    iterations: int = 2
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32, torch.float16, torch.bfloat16, torch.int64])

    def run(self, device, seed) -> dict[str, torch.Tensor]:
        if self.iterations < 1 or not self.shapes or any(len(shape) != 2 or min(shape) < 1 for shape in self.shapes):
            raise ValueError("positive iterations and nonempty positive 2D shapes are required")
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("shapes must be unique")
        check_dtypes(self.dtypes, (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int32, torch.int64))
        with SDCChecks(device, self, seed) as checks:
            device = checks.device
            for shape_index, (rows, columns) in enumerate(self.shapes):
                for iteration in range(self.iterations):
                    with progress(f"shape_{rows}x{columns}/iteration_{iteration:03d} ({iteration + 1}/{self.iterations})", "ITERATION"):
                        generator = cpu_generator(seed, shape_index * 1_000_033 + iteration * 1_000_003)
                        row_cpu = torch.randperm(rows, generator=generator, device="cpu")
                        col_cpu = torch.randperm(columns, generator=generator, device="cpu")
                        row_gpu, col_gpu = row_cpu.to(device), col_cpu.to(device)
                        for dtype in self.dtypes:
                            prefix = f"{str(dtype).removeprefix('torch.')}/{rows}x{columns}/iteration_{iteration:03d}"
                            if dtype.is_floating_point:
                                cpu = torch.randn((rows, columns), generator=generator, device="cpu").to(dtype)
                            else:
                                if dtype == torch.int32 and rows * columns >= 2**31:
                                    raise ValueError("int32 indexing payload would exceed its unique positive range")
                                cpu = torch.arange(rows * columns, dtype=dtype, device="cpu").reshape(rows, columns)
                            gpu = cpu.to(device)
                            checks.tensor(f"{prefix}/input", gpu, cpu)
                            selected = torch.index_select(gpu, 0, row_gpu)
                            checks.tensor(f"{prefix}/index_select", selected, cpu[row_cpu])
                            gathered = gpu.gather(1, col_gpu.unsqueeze(0).expand(rows, columns))
                            checks.tensor(f"{prefix}/gather", gathered, cpu[:, col_cpu])
                            scattered = torch.empty_like(gpu)
                            # A permutation writes each destination exactly once: no racing atomic sums.
                            scattered.scatter_(0, row_gpu.unsqueeze(1).expand(rows, columns), gpu)
                            expected = torch.empty_like(cpu)
                            expected[row_cpu] = cpu
                            checks.tensor(f"{prefix}/scatter", scattered, expected)
                            checks.tensor(f"{prefix}/transpose_copy", gpu.t().contiguous(), cpu.t().contiguous())
                            backing = torch.full((rows, columns * 2), 7, dtype=dtype, device=device)
                            strided = backing[:, ::2]
                            strided.copy_(gpu)
                            checks.tensor(f"{prefix}/strided_write", strided, cpu)
                            checks.tensor(f"{prefix}/strided_read_copy", strided.clone(), cpu)
                            checks.tensor(f"{prefix}/guard_lanes", backing[:, 1::2], torch.full_like(cpu, 7))
            return checks.finish()


EXPERIMENT = Indexing()

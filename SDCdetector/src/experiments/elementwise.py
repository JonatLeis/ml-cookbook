"""Tests elementwise floating-point math, casts, bounded chains and exact integer logic."""
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from reporting import progress
from experiments._utils import FLOAT_DTYPES, SDCChecks, ScreeningExperiment, check_dtypes, cpu_generator


@dataclass
class Elementwise(ScreeningExperiment):
    numel: int = 1024 * 1024
    iterations: int = 3
    chain_steps: int = 4
    dtypes: list[torch.dtype] = field(default_factory=lambda: list(FLOAT_DTYPES))
    integer_dtypes: tuple[torch.dtype, ...] = (torch.int32, torch.int64)

    def run(self, device, seed) -> dict[str, torch.Tensor]:
        if min(self.numel, self.iterations, self.chain_steps) < 1:
            raise ValueError("numel, iterations and chain_steps must be positive")
        check_dtypes(self.dtypes)
        check_dtypes(self.integer_dtypes, (torch.int32, torch.int64))
        float_ops = {
            "add": lambda x, y: x + y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y,
            "sqrt": lambda x, y: torch.sqrt(x.abs() + 0.25),
            "rsqrt": lambda x, y: torch.rsqrt(x.abs() + 0.25),
            "exp": lambda x, y: torch.exp(x),
            "log": lambda x, y: torch.log(x.abs() + 0.5),
            "sin": lambda x, y: torch.sin(x),
            "cos": lambda x, y: torch.cos(x),
            "sigmoid": lambda x, y: torch.sigmoid(x),
            "gelu": lambda x, y: F.gelu(x),
        }
        integer_ops = {
            "add": lambda x: x + 127,
            "multiply": lambda x: x * 17,
            "xor": lambda x: x ^ 0x5A5A,    # bitmask: 0101101001011010
            "and": lambda x: x & 0x55AA,    # bitmask: 0101010110101010
            "or": lambda x: x | 0xAA55,     # bitmask: 1010101001010101
            "left_shift": lambda x: x << 3,
            "right_shift": lambda x: x >> 2,
            "remainder": lambda x: x % 127,
        }
        with SDCChecks(device, self, seed) as checks:
            device = checks.device
            for iteration in range(self.iterations):
                with progress(f"iteration_{iteration:03d} ({iteration + 1}/{self.iterations})", "ITERATION"):
                    generator = cpu_generator(seed, iteration * 1_000_003)
                    x_cpu = torch.rand(self.numel, generator=generator, device="cpu") - 0.5
                    y_cpu = torch.rand(self.numel, generator=generator, device="cpu") + 0.5
                    for dtype in self.dtypes:
                        prefix = f"{str(dtype).removeprefix('torch.')}/iteration_{iteration:03d}"
                        x, y = x_cpu.to(dtype=dtype, device=device), y_cpu.to(dtype=dtype, device=device)
                        for name, operation in float_ops.items():
                            checks.tensor(f"{prefix}/{name}", operation(x, y))
                        for destination in self.dtypes:
                            checks.tensor(f"{prefix}/cast_to_{str(destination).removeprefix('torch.')}", x.to(destination))
                        chained = x
                        for step in range(self.chain_steps):
                            chained = torch.tanh(chained * 0.75 + y * 0.125)
                            checks.tensor(f"{prefix}/chain_{step:03d}", chained)

                    for dtype in self.integer_dtypes:
                        bits = 24 if dtype == torch.int32 else 48
                        # Keep every operation in range, including multiplication and shifts.
                        values = torch.arange(self.numel, dtype=torch.int64, device="cpu")
                        values = ((values * 0x9E3779 + (seed + iteration * 1_000_003) % (2**48))
                                  & ((1 << bits) - 1)).to(dtype)
                        gpu = values.to(device)
                        prefix = f"{str(dtype).removeprefix('torch.')}/iteration_{iteration:03d}"
                        for name, operation in integer_ops.items():
                            checks.tensor(f"{prefix}/{name}", operation(gpu), operation(values))
            return checks.finish()


EXPERIMENT = Elementwise()

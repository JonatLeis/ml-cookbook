"""Write, copy, read and invert exact 64-bit patterns across a fixed GPU allocation. Tests memory movements"""
from dataclasses import dataclass
import math
import time

import torch

from reporting import progress
from experiments._utils import SDCChecks, ScreeningExperiment, cpu_generator


@dataclass
class MemoryPatterns(ScreeningExperiment):
    memory_mib: int = 4096  # Total for BOTH buffers; explicit size stays stable across repeats.
    chunk_mib: int = 16
    iterations: int = 2
    retention_seconds: float = 0.0
    patterns: tuple[str, ...] = ("zeros", "ones", "alternating", "index", "walking_bit", "random")

    @staticmethod
    def _pattern(name, start, count, seed, iteration):
        if name == "zeros":
            return torch.zeros(count, dtype=torch.int64, device="cpu")
        if name == "ones":
            return torch.full((count,), -1, dtype=torch.int64, device="cpu")
        index = torch.arange(start, start + count, dtype=torch.int64, device="cpu")
        if name == "alternating":
            return torch.where((index + iteration) % 2 == 0, 0x5555555555555555, -6148914691236517206)
        if name == "index":
            return index ^ ((seed + iteration * 1_000_003) % (2**63))
        if name == "walking_bit":
            return torch.bitwise_left_shift(torch.ones_like(index), (index + iteration) % 64)
        if name == "random":
            generator = cpu_generator(seed, start + iteration * 1_000_003)
            low = torch.randint(0, 2**32, (count,), generator=generator, dtype=torch.int64, device="cpu")
            high = torch.randint(-(2**31), 2**31, (count,), generator=generator, dtype=torch.int64, device="cpu")
            return (high << 32) | low
        raise ValueError(f"unknown memory pattern: {name}")

    def run(self, device, seed) -> dict[str, torch.Tensor]:
        supported = {"zeros", "ones", "alternating", "index", "walking_bit", "random"}
        if (min(self.memory_mib, self.chunk_mib, self.iterations) < 1
                or not math.isfinite(self.retention_seconds) or self.retention_seconds < 0):
            raise ValueError("positive allocation, chunk size and iterations, and nonnegative retention are required")
        if not self.patterns or len(set(self.patterns)) != len(self.patterns) or set(self.patterns) - supported:
            raise ValueError(f"patterns must be a nonempty unique subset of {sorted(supported)}")
        words = self.memory_mib * 1024**2 // (2 * 8)
        chunk_words = self.chunk_mib * 1024**2 // 8
        with SDCChecks(device, self, seed) as checks:
            device = checks.device
            source = torch.empty(words, dtype=torch.int64, device=device)
            copied = torch.empty_like(source)
            for iteration in range(self.iterations):
                with progress(f"iteration_{iteration:03d} ({iteration + 1}/{self.iterations})", "ITERATION"):
                    for pattern in self.patterns:
                        prefix = f"iteration_{iteration:03d}/{pattern}"
                        # Fill the entire source before any readback so the test
                        # traverses the allocation rather than repeatedly testing one cache-sized block.
                        with progress(f"{prefix} memory_mib={self.memory_mib} chunk_mib={self.chunk_mib}", "PATTERN"):
                            for start in range(0, words, chunk_words):
                                count = min(chunk_words, words - start)
                                source[start:start + count].copy_(self._pattern(pattern, start, count, seed, iteration))
                            if self.retention_seconds:
                                time.sleep(self.retention_seconds)
                            copied.copy_(source)
                            for start in range(0, words, chunk_words):
                                count = min(chunk_words, words - start)
                                expected = self._pattern(pattern, start, count, seed, iteration)
                                checks.tensor(f"{prefix}/source_{start:012d}", source[start:start + count], expected)
                                checks.tensor(f"{prefix}/copy_{start:012d}", copied[start:start + count], expected)
                            source.bitwise_not_()
                            for start in range(0, words, chunk_words):
                                count = min(chunk_words, words - start)
                                expected = self._pattern(pattern, start, count, seed, iteration)
                                checks.tensor(f"{prefix}/inverted_{start:012d}", source[start:start + count], expected.bitwise_not())
                                checks.tensor(f"{prefix}/copy_guard_{start:012d}", copied[start:start + count], expected)
            return checks.finish()


EXPERIMENT = MemoryPatterns()

"""Small and quick workload for memory patterns experiments"""
from dataclasses import dataclass

from experiments.memory_patterns import MemoryPatterns


@dataclass
class MemoryPatternsQuick(MemoryPatterns):
    memory_mib: int = 384
    chunk_mib: int = 16
    iterations: int = 1
    retention_seconds: float = 0.0
    patterns: tuple[str, ...] = ('zeros', 'ones', 'alternating', 'index', 'walking_bit', 'random')


EXPERIMENT = MemoryPatternsQuick()

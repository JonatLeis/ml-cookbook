"""Interface for experiments"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from validation import assert_finite


@dataclass
class Experiment(ABC):
    """Subclasses own their configuration, computation and result checks."""

    @abstractmethod
    def run(self, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
        """Return named tensors; do not write files or change runner settings."""
        raise NotImplementedError

    def validate(self, results: dict[str, torch.Tensor]) -> None:
        """Override for additional checks and call super().validate(results)."""
        assert_finite(results)

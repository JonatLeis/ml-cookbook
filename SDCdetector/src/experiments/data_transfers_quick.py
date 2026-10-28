"""Small and quick workload for data transfer experiments"""
from dataclasses import dataclass

from experiments.data_transfers import Transfers


@dataclass
class TransfersQuick(Transfers):
    sizes: tuple[int, ...] = (256 * 1024,)
    iterations: int = 1


EXPERIMENT = TransfersQuick()

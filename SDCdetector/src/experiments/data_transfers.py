"""Data transfer experiments like exact host/device copies and rotating, bidirectional rank-to-rank payloads."""
from dataclasses import dataclass

import torch
import torch.distributed as dist

from reporting import progress
from experiments._utils import SDCChecks, ScreeningExperiment, cpu_generator


@dataclass
class Transfers(ScreeningExperiment):
    sizes: tuple[int, ...] = (4096, 1024 * 1024, 16 * 1024 * 1024)  # Bytes.
    iterations: int = 2

    @staticmethod
    def _payload(size, seed, rank, iteration, tag):
        # Generate a deterministic payload for a given rank, iteration, and tag. The payload is
        # seeded with the same value for all ranks, so that each rank can independently generate
        # the expected payload for any other rank. The payload is also seeded with the rank,
        # iteration, and tag, so that each rank can generate a unique payload for each transfer
        # and verify that the received payload matches the expected payload
        generator = cpu_generator(seed, rank * 1_000_003 + iteration * 1_000_033 + tag * 1_000_037)
        payload = torch.randint(0, 256, (size,), generator=generator, dtype=torch.uint8, device="cpu")
        # Source/round markers also detect stale or misrouted otherwise-valid data.
        header = rank.to_bytes(8, "little") + iteration.to_bytes(8, "little") + tag.to_bytes(8, "little")
        payload[:len(header)] = torch.tensor(list(header), dtype=torch.uint8, device="cpu")
        return payload

    def run(self, device, seed) -> dict[str, torch.Tensor]:
        if self.iterations < 1 or not self.sizes or min(self.sizes) < 32 or len(set(self.sizes)) != len(self.sizes):
            raise ValueError("positive iterations and unique transfer sizes of at least 32 bytes are required")
        with SDCChecks(device, self, seed) as checks:
            device = checks.device
            rank, world = checks.rank, checks.world
            if dist.get_backend() != "nccl": #assert nccl
                raise ValueError("CUDA GPU-to-GPU transfers require the runner's default NCCL process group")

            for size in self.sizes:
                for iteration in range(self.iterations):
                    prefix = f"bytes_{size}/iteration_{iteration:03d}"
                    with progress(f"{prefix} ({iteration + 1}/{self.iterations})", "ITERATION"):
                        expected = self._payload(size, seed, rank, iteration, 0)
                        host = expected.pin_memory()
                        send = torch.empty(size, dtype=torch.uint8, device=device)
                        send.copy_(host, non_blocking=True)
                        torch.cuda.synchronize(device)
                        checks.tensor(f"{prefix}/host_to_device", send, expected)
                        returned = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
                        returned.copy_(send, non_blocking=True)
                        torch.cuda.synchronize(device)
                        checks.tensor(f"{prefix}/device_to_host", returned, expected)

                        # Each offset forms a ring. Across offsets, each
                        # rank sends to and receives from every other rank.
                        for offset in range(1, world):
                            destination = (rank + offset) % world
                            source = (rank - offset) % world
                            with progress(f"{prefix}/offset_{offset} send={rank}->{destination} receive={source}->{rank}", "TRANSFER"):
                                outgoing = self._payload(size, seed, rank, iteration, offset)
                                send.copy_(outgoing)
                                checks.tensor(f"{prefix}/offset_{offset}/send_buffer", send, outgoing)
                                receive = torch.empty_like(send)
                                requests = dist.batch_isend_irecv([
                                    dist.P2POp(dist.isend, send, destination),
                                    dist.P2POp(dist.irecv, receive, source),
                                ])
                                for request in requests:
                                    request.wait()
                                torch.cuda.synchronize(device)
                                expected_receive = self._payload(size, seed, source, iteration, offset)
                                checks.tensor(f"{prefix}/offset_{offset}/received", receive, expected_receive,
                                              owners=(rank, source))
            return checks.finish()


EXPERIMENT = Transfers()

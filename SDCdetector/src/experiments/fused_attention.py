"""Explicit CUDA FlashAttention forward/backward. SDPA kernel"""
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from reporting import progress
from experiments._utils import SDCChecks, ScreeningExperiment, check_dtypes, cpu_generator


@dataclass
class FusedAttention(ScreeningExperiment):
    batch_size: int = 2
    heads: int = 4
    lengths: tuple[int, ...] = (128, 512)
    head_dims: tuple[int, ...] = (64, 128)
    causal_modes: tuple[bool, ...] = (False, True)
    iterations: int = 2
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float16, torch.bfloat16])

    def run(self, device, seed) -> dict[str, torch.Tensor]:
        if torch.device(device).type != "cuda":
            raise ValueError("this experiment requires CUDA FlashAttention; CPU execution is not this test")
        if min(self.batch_size, self.heads, self.iterations) < 1:
            raise ValueError("batch_size, heads and iterations must be positive")
        if (not self.lengths or not self.head_dims or not self.causal_modes
                or min(self.lengths) < 1 or min(self.head_dims) < 1
                or len(set(self.lengths)) != len(self.lengths)
                or len(set(self.head_dims)) != len(self.head_dims)
                or len(set(self.causal_modes)) != len(self.causal_modes)
                or any(type(value) is not bool for value in self.causal_modes)):
            raise ValueError("lengths/head_dims must be positive and unique; causal_modes must be unique booleans")
        check_dtypes(self.dtypes, (torch.float16, torch.bfloat16))
        with SDCChecks(device, self, seed) as checks:
            device = checks.device
            for dtype in self.dtypes:
                for length in self.lengths:
                    for head_dim in self.head_dims:
                        for causal in self.causal_modes:
                            for iteration in range(self.iterations):
                                prefix = (f"{str(dtype).removeprefix('torch.')}/length_{length}/head_dim_{head_dim}"
                                          f"/causal_{int(causal)}/iteration_{iteration:03d}")
                                with progress(f"{prefix} ({iteration + 1}/{self.iterations})", "ITERATION"):
                                    shape = (self.batch_size, self.heads, length, head_dim)
                                    generator = cpu_generator(seed, length * 1009 + head_dim * 101 + iteration * 1_000_003)
                                    q, k, v = [(torch.randn(shape, generator=generator, device="cpu") * 0.25).to(
                                        dtype=dtype, device=device).requires_grad_() for _ in range(3)]
                                    # A missing/unsupported Flash kernel is an execution error,
                                    # not a passing test on a different backend.
                                    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                                        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
                                        checks.tensor(f"{prefix}/forward", output)
                                        upstream = (torch.randn(shape, generator=generator, device="cpu") * 0.125).to(
                                            dtype=dtype, device=device)
                                        gradients = torch.autograd.grad(output, (q, k, v), grad_outputs=upstream)
                                    for name, gradient in zip(("dq", "dk", "dv"), gradients):
                                        checks.tensor(f"{prefix}/{name}", gradient)
            return checks.finish()


EXPERIMENT = FusedAttention()

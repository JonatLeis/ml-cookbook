"""GPU-local reductions and normalization, including deterministic backward paths."""
from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F

from reporting import progress
from experiments._utils import FLOAT_DTYPES, SDCChecks, ScreeningExperiment, check_dtypes, cpu_generator


@dataclass
class LocalReductions(ScreeningExperiment):
    rows: int = 64
    widths: tuple[int, ...] = (127, 1024, 4097)
    iterations: int = 3
    eps: float = 1e-5
    dtypes: list[torch.dtype] = field(default_factory=lambda: list(FLOAT_DTYPES))

    def run(self, device, seed) -> dict[str, torch.Tensor]:
        if self.rows < 1 or self.iterations < 1 or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("rows, iterations and eps must be positive")
        if not self.widths or min(self.widths) < 1 or len(set(self.widths)) != len(self.widths):
            raise ValueError("widths must be positive, nonempty and unique")
        check_dtypes(self.dtypes)
        with SDCChecks(device, self, seed) as checks:
            device = checks.device
            for dtype in self.dtypes:
                for width in self.widths:
                    for iteration in range(self.iterations):
                        prefix = f"{str(dtype).removeprefix('torch.')}/width_{width}/iteration_{iteration:03d}"
                        with progress(f"{prefix} ({iteration + 1}/{self.iterations})", "ITERATION"):
                            generator = cpu_generator(seed, width * 1009 + iteration * 1_000_003)
                            x_cpu = torch.randn((self.rows, width), generator=generator, device="cpu") * 0.25
                            y = (torch.randn((self.rows, width), generator=generator, device="cpu") * 0.25).to(
                                dtype=dtype, device=device)
                            for operation in ("sum", "square_sum", "dot", "softmax", "layer_norm", "rms_norm"):
                                with progress(f"{prefix}/{operation}", "OPERATION"):
                                    x = x_cpu.to(dtype=dtype, device=device).requires_grad_()
                                    weight = torch.ones(width, dtype=dtype, device=device, requires_grad=True)
                                    bias = torch.zeros(width, dtype=dtype, device=device, requires_grad=True)
                                    inputs, names = [x], ["input_gradient"]
                                    if operation == "sum":
                                        output = x.sum(dim=-1)
                                    elif operation == "square_sum":
                                        output = x.square().sum(dim=-1)
                                    elif operation == "dot":
                                        output = (x * y).sum(dim=-1)
                                    elif operation == "softmax":
                                        output = x.softmax(dim=-1)
                                    elif operation == "layer_norm":
                                        output = F.layer_norm(x, (width,), weight, bias, eps=self.eps)
                                        inputs += [weight, bias]
                                        names += ["weight_gradient", "bias_gradient"]
                                    else:
                                        output = F.rms_norm(x, (width,), weight, eps=self.eps)
                                        inputs += [weight]
                                        names += ["weight_gradient"]
                                    checks.tensor(f"{prefix}/{operation}/forward", output)
                                    # Random upstream gradients avoid the trivial zero gradient
                                    # of sum(softmax(x)) and exercise nontrivial backward work.
                                    upstream = (torch.rand(output.shape, generator=generator, device="cpu") - 0.5).to(
                                        dtype=dtype, device=device)
                                    gradients = torch.autograd.grad(output, inputs, grad_outputs=upstream)
                                    for name, gradient in zip(names, gradients):
                                        checks.tensor(f"{prefix}/{operation}/{name}", gradient)
            return checks.finish()


EXPERIMENT = LocalReductions()

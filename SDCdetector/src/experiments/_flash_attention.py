"""Shared library for FlashAttention experiments"""
from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
from typing import ClassVar

import torch
import torch.distributed as dist

from experiments._utils import SDCChecks, ScreeningExperiment, check_dtypes, cpu_generator
from reporting import diagnostic, progress


def _call_backend(function, q, k, v, name, options):
    output = function(q.contiguous(), k.contiguous(), v.contiguous(), **options)
    if isinstance(output, tuple) and output:
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"{name} must return a Tensor or (Tensor, auxiliary_data)")
    if output.shape != q.shape or output.dtype != q.dtype or output.device != q.device:
        raise ValueError(f"{name} output must match Q's shape, dtype and device")
    return output


def _cpu_data(shape, dtype, pattern, scale, generator):
    if pattern == "normal":
        data = torch.randn(shape, generator=generator, dtype=torch.float32)
    elif pattern == "uniform":
        data = torch.rand(shape, generator=generator, dtype=torch.float32) * 2 - 1
    elif pattern == "zeros":
        data = torch.zeros(shape, dtype=torch.float32)
    else:
        raise ValueError(f"unknown data pattern: {pattern}")
    return (data * scale).to(dtype).contiguous()


@dataclass
class FlashAttention(ScreeningExperiment):
    name: ClassVar[str]
    backend: ClassVar[str]
    shapes: tuple[tuple[int, int, int, int], ...] = (
        (1, 1, 1, 64), (1, 128, 2, 64), (1, 129, 2, 64), (2, 256, 4, 64),
        (2, 512, 4, 64), (1, 257, 2, 96), (2, 256, 4, 128), (1, 128, 2, 256))
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float16, torch.bfloat16])
    causal_modes: tuple[bool, ...] = (False, True)
    data_patterns: tuple[str, ...] = ("normal",)
    input_scale: float = 0.25
    iterations: int = 1
    backward: bool = True

    @abstractmethod
    def _load_backend(self):
        """Import only this experiment's installed backend."""
        raise NotImplementedError

    def _options(self, causal):
        options = {"causal": causal}
        if self.backward:
            # Missing deterministic support is an execution error, never a fallback.
            options["deterministic"] = True
        return options

    def _backward_context(self, device):
        return None

    def _check_config(self):
        if (not self.shapes or len(set(self.shapes)) != len(self.shapes)
                or any(len(shape) != 4 or any(type(size) is not int or size < 1 for size in shape)
                       for shape in self.shapes)):
            raise ValueError("shapes must be unique positive [batch, sequence, heads, head_dim] tuples")
        check_dtypes(self.dtypes, (torch.float16, torch.bfloat16))
        if (not self.causal_modes or len(set(self.causal_modes)) != len(self.causal_modes)
                or any(type(mode) is not bool for mode in self.causal_modes)):
            raise ValueError("causal_modes must be nonempty unique booleans")
        if (not self.data_patterns or len(set(self.data_patterns)) != len(self.data_patterns)
                or any(pattern not in ("normal", "uniform", "zeros") for pattern in self.data_patterns)):
            raise ValueError("data_patterns must be a nonempty unique subset of [normal, uniform, zeros]")
        if not math.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError("input_scale must be finite and positive")
        if type(self.iterations) is not int or self.iterations < 1 or type(self.backward) is not bool:
            raise ValueError("iterations must be positive and backward must be boolean")

    def _inputs(self, checks, shape, dtype, pattern, seed, case):
        generator = cpu_generator(seed)
        tensors = []
        for index in range(4 if self.backward else 3):
            # The upstream gradient stays nonzero/random even for the zero-input case.
            tensor = checks.call(
                case, f"create_input_{index}",
                lambda: (_cpu_data(shape, dtype, pattern if index < 3 else "normal",
                                   self.input_scale if index < 3 else 0.125, generator)
                         if checks.rank == 0 else torch.empty(shape, dtype=dtype, device="cpu")))
            dist.broadcast(tensor.view(torch.uint8), src=0, group=checks.control)
            tensors.append(tensor)
        return tensors

    def _compute(self, function, inputs, device, causal, backward_context=None):
        with torch.inference_mode(False), torch.enable_grad():
            q, k, v = [tensor.to(device).clone().requires_grad_(self.backward) for tensor in inputs[:3]]
            output = _call_backend(function, q, k, v, self.name, self._options(causal))
            gradients = ()
            if self.backward:
                with backward_context() if backward_context is not None else nullcontext():
                    gradients = torch.autograd.grad(output, (q, k, v), grad_outputs=inputs[3].to(device))
            # Finish GPU readback before the error exchange and replica collectives.
            return ([tensor.detach().cpu().clone() for tensor in (q, k, v)],
                    output.detach().cpu().clone(),
                    [gradient.detach().cpu().clone() for gradient in gradients])

    def run(self, device, seed):
        self._check_config()
        device = torch.device(device)
        if device.type != "cuda" or torch.version.hip:
            raise ValueError(f"{self.name} requires NVIDIA CUDA GPUs")
        with SDCChecks(device, self, seed) as checks:
            function = checks.call(self.name, "load", self._load_backend)
            backward_context = None
            if self.backward:
                backward_context = checks.call(
                    self.name, "backward_setup", lambda: self._backward_context(checks.device))
            diagnostic("PLACEMENT", experiment=self.name, backend=self.backend,
                       devices=checks.devices)
            for dtype in self.dtypes:
                for shape_index, shape in enumerate(self.shapes):
                    for pattern_index, pattern in enumerate(self.data_patterns):
                        for iteration in range(self.iterations):
                            case_seed = (seed + shape_index * 1_000_033 + pattern_index * 1_000_037
                                         + iteration * 1_000_003) % (2**63)
                            prefix = (f"{str(dtype).removeprefix('torch.')}/{'x'.join(map(str, shape))}"
                                      f"/{pattern}/iteration_{iteration:03d}")
                            inputs = self._inputs(checks, shape, dtype, pattern, case_seed, prefix)
                            for name, tensor in zip(("q", "k", "v", "upstream"), inputs):
                                checks.tensor(f"{prefix}/shared_{name}", tensor)
                            for causal in self.causal_modes:
                                case = f"{self.name}/{prefix}/causal_{int(causal)}"
                                with progress(case):
                                    copied, output, gradients = checks.call(
                                        case, "compute",
                                        lambda: self._compute(function, inputs, checks.device,
                                                              causal, backward_context))
                                    for name, actual, expected in zip(("q", "k", "v"), copied, inputs):
                                        checks.tensor(f"{case}/input_{name}", actual, expected)
                                    checks.tensor(f"{case}/forward", output)
                                    for name, gradient in zip(("dq", "dk", "dv"), gradients):
                                        checks.tensor(f"{case}/{name}", gradient)
            return checks.finish()

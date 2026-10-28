"""Independent TP=2 transformer training: snapshot training stages and compare replicas."""
from dataclasses import dataclass, field
from datetime import timedelta
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from reporting import diagnostic, record_test
from experiment import Experiment
from experiments._utils import (
    FLOAT_DTYPES, candidate_ranks, cpu_generator, device_info, equivalence_classes, tensor_bytes,
)
from validation import ValidationError


SHARDED_PARAMETERS = frozenset({
    "qkv.weight", "qkv.bias", "projection.weight",
    "mlp.0.weight", "mlp.0.bias", "mlp.2.weight",
})


class _CopyToTP(torch.autograd.Function):
    """Replicated input -> local computation; sum input gradients in backward."""

    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        total = gradient.contiguous().clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=ctx.group)
        return total, None


class _SumFromTP(torch.autograd.Function):
    """Sum partial outputs in forward; each shard receives its own gradient."""

    @staticmethod
    def forward(ctx, value, group):
        total = value.contiguous().clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
        return total

    @staticmethod
    def backward(ctx, gradient):
        # Do not all-reduce here: that would multiply the gradient by TP size.
        return gradient, None


class TinyTransformer(nn.Module):
    """The original block, with explicit head and MLP sharding for TP=2."""

    def __init__(self, config, tp_rank=None, tp_group=None):
        super().__init__()
        width = config.width
        self.head_dim = width // config.heads
        self.heads = config.heads
        self.tp_group = tp_group
        self.tokens = nn.Embedding(config.vocab_size, width)
        self.positions = nn.Embedding(config.length, width)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, config.vocab_size, bias=False)
        if tp_rank is not None:
            self._shard(width, tp_rank)

    def _shard(self, width, tp_rank):
        if tp_rank not in (0, 1) or self.tp_group is None:
            raise ValueError("TP requires shard 0 or 1 and a two-rank process group")
        half = width // 2
        start, end = tp_rank * half, (tp_rank + 1) * half

        def parameter(value):
            return nn.Parameter(value.detach().clone().contiguous())

        # Each rank needs half of Q, half of K AND half of V. Taking a single
        # contiguous half of the fused QKV matrix would split these incorrectly.
        self.qkv.weight = parameter(
            self.qkv.weight.reshape(3, width, width)[:, start:end].reshape(3 * half, width))
        self.qkv.bias = parameter(
            self.qkv.bias.reshape(3, width)[:, start:end].reshape(3 * half))
        self.qkv.out_features = 3 * half
        self.projection.weight = parameter(self.projection.weight[:, start:end])
        self.projection.in_features = half

        start, end = tp_rank * (2 * width), (tp_rank + 1) * (2 * width)
        self.mlp[0].weight = parameter(self.mlp[0].weight[start:end])
        self.mlp[0].bias = parameter(self.mlp[0].bias[start:end])
        self.mlp[0].out_features = 2 * width
        self.mlp[2].weight = parameter(self.mlp[2].weight[:, start:end])
        self.mlp[2].in_features = 2 * width
        self.heads //= 2

    def _copy(self, value):
        return value if self.tp_group is None else _CopyToTP.apply(value, self.tp_group)

    def _row_linear(self, value, layer):
        partial = F.linear(value, layer.weight, bias=None)
        if self.tp_group is not None:
            partial = _SumFromTP.apply(partial, self.tp_group)
        # A replicated bias is added once AFTER the sum, on each rank.
        return partial + layer.bias

    def forward(self, tokens):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        qkv = self.qkv(self._copy(self.norm1(x)))
        qkv = qkv.reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mixed = scores.masked_fill(mask, float("-inf")).softmax(-1) @ v
        mixed = mixed.transpose(1, 2).contiguous().reshape(
            batch, length, self.heads * self.head_dim)
        x = x + self._row_linear(mixed, self.projection)
        hidden = self.mlp[1](self.mlp[0](self._copy(self.norm2(x))))
        x = x + self._row_linear(hidden, self.mlp[2])
        return self.output(self.final_norm(x))


def _log_mismatch(key, tensor, values, classes, devices, scope, tp_shard=None):
    involved = sorted({rank // 2 for members in classes for rank in members})
    if scope == "within_tp":
        suspects = involved
        attribution = "replicated_values_disagree_within_pair"
    else:
        ranks, attribution = candidate_ranks(classes)
        suspects = sorted({rank // 2 for rank in ranks}) if attribution == "majority_outlier" else []
    first, other = classes[0][0], classes[1][0]
    offsets = torch.nonzero(values[first] != values[other], as_tuple=False)
    diagnostic("MISMATCH", key=key, dtype=str(tensor.dtype), shape=list(tensor.shape),
               scope=scope, tp_shard=tp_shard, attribution=attribution,
               suspect_tp_groups=suspects, equivalence_classes=classes,
               first_differing_element=int(offsets[0, 0]) // tensor.element_size(),
               tp_groups=[{"id": group, "gpus": devices[2 * group:2 * group + 2]}
                          for group in involved])


def _compare_replicas(results, sharded_keys, devices, control, local_errors):
    """All ranks participate; rank 0 compares one saved tensor at a time."""
    rank, world = dist.get_rank(), dist.get_world_size()
    schema = [(key, str(tensor.dtype), tuple(tensor.shape), key in sharded_keys)
              for key, tensor in results.items()]
    reports = [None] * world
    dist.all_gather_object(reports, {"schema": schema, "errors": local_errors}, group=control)
    same_schema = all(report["schema"] == reports[0]["schema"] for report in reports)
    passed = same_schema and not any(report["errors"] for report in reports)
    if rank == 0:
        for index, report in enumerate(reports):
            for error in report["errors"]:
                diagnostic("INVALID_TRAINING_STATE", error=error,
                           tp_group=index // 2, gpu=devices[index],
                           group_gpus=devices[2 * (index // 2):2 * (index // 2) + 2])
        if not same_schema:
            diagnostic("SCHEMA_MISMATCH", devices=devices,
                       schemas=[report["schema"] for report in reports])
    if same_schema:
        for key in results:
            value = tensor_bytes(results[key])
            gathered = [torch.empty_like(value) for _ in range(world)] if rank == 0 else None
            dist.gather(value, gather_list=gathered, dst=0, group=control)
            if rank != 0:
                continue
            # Compare corresponding shards across independent model copies.
            for shard in (0, 1):
                classes = equivalence_classes(gathered, range(shard, world, 2))
                if len(classes) > 1:
                    passed = False
                    _log_mismatch(key, results[key], gathered, classes, devices,
                                  "replicas", tp_shard=shard)
            # Unsharded tensors must also agree between the two TP ranks.
            if key not in sharded_keys:
                for first in range(0, world, 2):
                    if not torch.equal(gathered[first], gathered[first + 1]):
                        passed = False
                        _log_mismatch(key, results[key], gathered,
                                      [[first], [first + 1]], devices, "within_tp")
    status = torch.tensor(int(passed), dtype=torch.uint8)
    dist.broadcast(status, src=0, group=control)
    if rank == 0:
        diagnostic("PASS" if status.item() else "FAIL", replicas=world // 2,
                   saved_tensors_per_rank=len(results), comparison="exact_bytes")
    record_test("tp_replica_state", "PASS" if status.item() else "FAIL")
    return status


@dataclass
class TransformerTraining(Experiment):
    steps: int = 5
    batch_size: int = 2
    length: int = 16
    vocab_size: int = 64
    width: int = 32
    heads: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    adam_eps: float = 1e-4  # 1e-8 would round to zero in native FP16 AdamW.
    dtypes: list[torch.dtype] = field(default_factory=lambda: list(FLOAT_DTYPES))

    def _check_config(self):
        if min(self.steps, self.batch_size, self.length, self.width, self.heads) < 1:
            raise ValueError("steps, batch_size, length, width and heads must be positive")
        if self.vocab_size < 2 or self.width % self.heads or self.heads % 2:
            raise ValueError("vocab_size must be >= 2; width divisible by heads; heads divisible by TP=2")
        if not self.dtypes or len(set(self.dtypes)) != len(self.dtypes):
            raise ValueError("dtypes must be a nonempty list without duplicates")
        unsupported = [str(dtype) for dtype in self.dtypes if dtype not in FLOAT_DTYPES]
        if unsupported:
            raise ValueError(f"unsupported training dtypes: {unsupported}; supported: {FLOAT_DTYPES}")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.adam_eps) or self.adam_eps <= 0:
            raise ValueError("adam_eps must be finite and positive")
        if torch.float16 in self.dtypes and self.adam_eps < torch.finfo(torch.float16).tiny:
            raise ValueError("native FP16 AdamW needs adam_eps >= torch.finfo(torch.float16).tiny")

    def _run_dtype(self, device, seed, dtype, tp_group, results, sharded_keys):
        tag = str(dtype).removeprefix("torch.")

        def save(key, tensor, name=None):
            key = f"{tag}/{key}"
            results[key] = tensor.detach().cpu().clone()
            if name in SHARDED_PARAMETERS:
                sharded_keys.add(key)

        generator = cpu_generator(seed)
        sequence = torch.randint(self.vocab_size, (self.batch_size, self.length + 1),
                                 generator=generator)
        tokens, targets = sequence[:, :-1].to(device), sequence[:, 1:].to(device)
        torch.manual_seed(seed)
        # Every rank initializes the same full CPU model, then keeps its own
        # shard. There is no initialization broadcast between model copies.
        model = TinyTransformer(self, dist.get_rank() % 2, tp_group).to(
            device=device, dtype=dtype).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay, eps=self.adam_eps,
                                      foreach=False, fused=False)
        save("tokens", tokens)
        save("targets", targets)
        for name, parameter in model.named_parameters():
            save(f"initial/{name}", parameter, name)
        for step in range(1, self.steps + 1):
            prefix = f"step_{step:03d}"
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss_input = logits.float() if dtype in (torch.float16, torch.bfloat16) else logits
            loss = F.cross_entropy(loss_input.reshape(-1, self.vocab_size), targets.reshape(-1))
            # Snapshot before backward: later stages cannot change this evidence.
            save(f"{prefix}/forward/logits", logits)
            save(f"{prefix}/forward/loss", loss)
            loss.backward()
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                # Preserve a failed gradient as evidence without making just
                # one rank exit before the coordinated comparison.
                if gradient is None:
                    gradient = torch.full_like(parameter, float("nan"))
                save(f"{prefix}/backward/gradient/{name}", gradient, name)
            optimizer.step()
            for name, parameter in model.named_parameters():
                save(f"{prefix}/update/parameter/{name}", parameter, name)
                for state_field in ("step", "exp_avg", "exp_avg_sq"):
                    state = optimizer.state[parameter].get(state_field)
                    if state is None:
                        state = torch.tensor(float("nan"))
                    save(f"{prefix}/update/optimizer/{name}/{state_field}", state,
                         None if state_field == "step" else name)

    def _training_errors(self, results):
        errors = []
        for key, tensor in results.items():
            if not torch.isfinite(tensor).all().item():
                errors.append(f"{key}: non-finite tensor (or missing gradient/optimizer state)")
        for dtype in self.dtypes:
            tag = str(dtype).removeprefix("torch.")
            initial = f"{tag}/initial/"
            names = [key.removeprefix(initial) for key in results if key.startswith(initial)]
            if not names:
                errors.append(f"{tag}: missing initial parameters")
                continue
            for step in range(1, self.steps + 1):
                prefix = f"{tag}/step_{step:03d}"
                previous = f"{tag}/initial" if step == 1 else f"{tag}/step_{step - 1:03d}/update/parameter"
                try:
                    if not any(torch.count_nonzero(results[f"{prefix}/backward/gradient/{name}"]).item()
                               for name in names):
                        errors.append(f"{prefix}/backward: all gradients are zero")
                    if not any(not torch.equal(results[f"{previous}/{name}"],
                                               results[f"{prefix}/update/parameter/{name}"])
                               for name in names):
                        errors.append(f"{prefix}/update: parameters did not change")
                    for name in names:
                        if results[f"{prefix}/update/optimizer/{name}/step"].item() != step:
                            errors.append(f"{prefix}/update: incorrect AdamW step for {name}")
                except KeyError as exc:
                    errors.append(f"{prefix}: missing result {exc}")
        return errors

    def run(self, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
        self._check_config()
        if not dist.is_initialized():
            raise ValueError("initialize torch.distributed first; launch with torchrun")
        world, rank = dist.get_world_size(), dist.get_rank()
        if world < 4 or world % 2:
            raise ValueError("need an even world_size >= 4: at least two independent TP=2 replicas")
        if not dist.is_gloo_available():
            raise ValueError("Gloo is required for the independent CPU snapshot comparison")
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise ValueError("runner must enable torch.use_deterministic_algorithms(True, warn_only=False)")
        device = torch.device(device)
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        tp_group = None
        try:
            devices = [None] * world
            info = {**device_info(device, rank), "tp_group": rank // 2, "tp_shard": rank % 2}
            dist.all_gather_object(devices, info, group=control)
            if len({entry["host"] for entry in devices}) != 1:
                raise ValueError("this experiment currently requires all ranks on one node")
            if device.type == "cuda":
                uuids = [entry["uuid"] for entry in devices]
                if None in uuids or len(set(uuids)) != world:
                    raise ValueError("need one distinct GPU UUID per rank; this PyTorch build must expose device.uuid")
                if len({entry["gpu_model"] for entry in devices}) != 1:
                    raise ValueError("exact replica comparison requires the same GPU model on all ranks")
            if rank == 0:
                diagnostic("PLACEMENT", tp_size=2, replicas=world // 2, devices=devices)
            # All WORLD ranks must call new_group in the same order, including
            # ranks that are not members of the respective pair.
            for first in range(0, world, 2):
                group = dist.new_group(ranks=[first, first + 1], timeout=timedelta(seconds=120))
                if rank in (first, first + 1):
                    tp_group = group
            results, sharded_keys = {}, set()
            for dtype in self.dtypes:
                if rank == 0:
                    diagnostic("DTYPE_START", dtype=str(dtype), steps=self.steps)
                self._run_dtype(device, seed, dtype, tp_group, results, sharded_keys)
            local_errors = self._training_errors(results)
            results["checks/distributed_ok"] = _compare_replicas(
                results, sharded_keys, devices, control, local_errors)
            # Return normally even on a detected mismatch. The runner saves
            # results.pt, then validate() raises on ALL ranks using this flag.
            return results
        finally:
            if tp_group is not None:
                dist.destroy_process_group(tp_group)
            dist.destroy_process_group(control)

    def validate(self, results: dict[str, torch.Tensor]) -> None:
        # No collectives: the runner also validates saved references offline.
        super().validate(results)
        flag = results.get("checks/distributed_ok")
        if flag is None or flag.numel() != 1 or flag.item() != 1:
            raise ValidationError("TP replica/state validation failed; see report failures on rank 0")
        errors = self._training_errors(results)
        if errors:
            raise ValidationError("; ".join(errors))


# EXPERIMENT = TransformerTraining()

EXPERIMENT = TransformerTraining(
    steps=5,
    batch_size=2,
    length=16,
    vocab_size=64,
    width=32,
    heads=2,
    dtypes=[torch.float32, torch.float16, torch.bfloat16, torch.float64],
)

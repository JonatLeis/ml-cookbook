"""Shared utilities for experiments, including SDCChecks and validation"""
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import socket
from uuid import UUID

import torch
import torch.distributed as dist

from experiment import Experiment
from reporting import debug, diagnostic, progress, record_test
from validation import ValidationError


FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
NCCL_ALLREDUCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32,
                       torch.float64, torch.int32, torch.int64)


def check_dtypes(dtypes, supported=FLOAT_DTYPES):
    if not dtypes or len(set(dtypes)) != len(dtypes):
        raise ValueError("dtypes must be nonempty and unique")
    if any(dtype not in supported for dtype in dtypes):
        raise ValueError(f"supported dtypes are {supported}; got {dtypes}")


def tensor_bytes(tensor):
    """Logical bytes on CPU, including signed zero and noncontiguous inputs."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def log_once(message):
    if not dist.is_initialized() or dist.get_rank() == 0:
        debug(message)


def gather_rank_tensors(tensor):
    """Gather equal-shaped tensors using the default process group, then read back."""
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return {f"rank_{rank:04d}": value.cpu() for rank, value in enumerate(gathered)}


def generate_shared_random_tensor(shape, device, seed):
    """Broadcast seeded data and collectively reject any corrupted receive."""
    rank = dist.get_rank()
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(seed)
        tensor = torch.randn(shape, device=device, generator=generator)
        digest = [hashlib.sha256(tensor_bytes(tensor).numpy().tobytes()).hexdigest()]
    else:
        tensor = torch.empty(shape, device=device)
        digest = [None]
    dist.broadcast(tensor, src=0)
    dist.broadcast_object_list(digest, src=0)
    matches = hashlib.sha256(tensor_bytes(tensor).numpy().tobytes()).hexdigest() == digest[0]
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, matches)
    if not all(reports):
        failed = [rank for rank, matches in enumerate(reports) if not matches]
        raise ValidationError(f"shared tensor differs from source on ranks {failed}")
    return tensor


def cpu_generator(seed, offset=0):
    """Create an independent CPU RNG without changing the caller's RNG state."""
    return torch.Generator(device="cpu").manual_seed((seed + offset) % (2**63))


def device_info(device, rank=None):
    """Return consistent placement metadata for comparison and failure reports."""
    rank = dist.get_rank() if rank is None else rank
    info = {"rank": rank, "host": socket.gethostname(), "device": str(device),
            "uuid": None, "gpu_model": None}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        raw = getattr(properties, "uuid", None)
        if isinstance(raw, bytes) and len(raw) == 16:
            raw = "GPU-" + str(UUID(bytes=raw))
        info.update(uuid=str(raw) if raw is not None else None, gpu_model=properties.name)
    return info


def equivalence_classes(values, ranks=None):
    """Group ranks whose supplied tensors have identical values."""
    classes = []
    ranks = range(len(values)) if ranks is None else ranks
    for rank in ranks:
        value = values[rank]
        for members in classes:
            if torch.equal(value, values[members[0]]):
                members.append(rank)
                break
        else:
            classes.append([rank])
    return classes


def candidate_ranks(classes):
    """Attribute differences only with a strict majority; otherwise implicate all."""
    largest = max(classes, key=len)
    majority = len(largest) > sum(map(len, classes)) / 2
    ranks = [rank for members in classes if not majority or members is not largest for rank in members]
    return ranks, "majority_outlier" if majority else "ambiguous_no_majority"


class SDCChecks:
    """Check every supplied tensor now; return small, repeatable tensor results.

    With expected=..., each rank checks its full tensor against that exact
    reference. Otherwise, rank 0 compares full byte chunks across GPU replicas.
    SHA-256 is retained for the existing runner's repetition/reference checks;
    current-run screening uses actual bytes, not just a checksum.
    """

    def __init__(self, device, experiment, seed, chunk_bytes=1024 * 1024):
        self.device = torch.device(device)
        self.experiment = type(experiment).__name__
        self.config = json.loads(json.dumps(asdict(experiment), default=str))
        self.seed = seed
        self.chunk_bytes = chunk_bytes
        self._results = {}
        self._count = 0
        self.control = None

    def _log(self, event, **fields):
        if self.rank == 0:
            diagnostic(event, experiment=self.experiment, **fields)

    def __enter__(self):
        if not dist.is_initialized() or dist.get_world_size() < 2:
            raise ValueError("launch with torchrun and at least two initialized ranks")
        if not dist.is_gloo_available():
            raise ValueError("Gloo is required for CPU validation/control traffic")
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
            raise ValueError("runner must enable deterministic algorithms with warn_only=False")
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("only CUDA screening and CPU/Gloo logic checks are supported")
        if self.chunk_bytes < 8:
            raise ValueError("chunk_bytes must be at least eight")
        if self.device.type == "cuda":
            index = self.device.index
            self.device = torch.device("cuda", torch.cuda.current_device() if index is None else index)
            torch.cuda.set_device(self.device)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        try:
            headers = [None] * self.world
            dist.all_gather_object(headers, {"device": device_info(self.device, self.rank),
                                             "config": self.config, "experiment": self.experiment,
                                             "chunk_bytes": self.chunk_bytes, "seed": self.seed},
                                   group=self.control)
            if any(item["config"] != self.config or item["experiment"] != self.experiment
                   or item["chunk_bytes"] != self.chunk_bytes or item["seed"] != self.seed for item in headers):
                raise ValueError("all ranks must run the same experiment, configuration and seed")
            self.devices = [item["device"] for item in headers]
            if len({item["device"].split(":")[0] for item in self.devices}) != 1:
                raise ValueError("all ranks must use the same device type")
            if self.device.type == "cuda":
                uuids = [item["uuid"] for item in self.devices]
                if None in uuids or len(set(uuids)) != self.world:
                    raise ValueError("each rank must expose a distinct CUDA GPU UUID")
                if len({item["gpu_model"] for item in self.devices}) != 1:
                    raise ValueError("exact floating-point replica checks require the same GPU model")
            self._candidates = torch.zeros(self.world, dtype=torch.uint8, device="cpu")
            self._log("PLACEMENT", devices=self.devices, config=self.config)
            return self
        except BaseException:
            dist.destroy_process_group(self.control)
            self.control = None
            raise

    def __exit__(self, exc_type, exc, traceback):
        if self.control is not None:
            dist.destroy_process_group(self.control)
            self.control = None

    def call(self, case, stage, function):
        """Exchange rank-local execution errors before peers enter result collectives."""
        result, error = None, None
        try:
            result = function()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        reports = [None] * self.world
        dist.all_gather_object(reports, error, group=self.control)
        errors = [{"device": self.devices[rank], "error": message}
                  for rank, message in enumerate(reports) if message is not None]
        if errors:
            record_test(f"{case}/{stage}", "ERROR", errors=errors)
            raise RuntimeError(f"{case}/{stage}: execution failed on ranks: {errors}")
        return result

    def tensor(self, name, actual, expected=None, owners=None):
        """All ranks call in the same order, before actual can be overwritten.

        owners optionally identifies both endpoints of a transfer. A mismatch
        then marks those endpoints as a candidate set, without assigning blame.
        """
        with progress(name):
            owners = tuple(owners) if owners is not None else (self.rank,)
            local_header = {
                "schema": (name, str(actual.dtype), tuple(actual.shape), expected is not None),
                "expected_schema": None if expected is None else (str(expected.dtype), tuple(expected.shape)),
                "owners": owners,
            }
            headers = [None] * self.world
            dist.all_gather_object(headers, local_header, group=self.control)
            schema = local_header["schema"]
            if (name + "/ok" in self._results or actual.numel() < 1
                    or any(item["schema"] != schema for item in headers)
                    or any(not item["owners"] or any(rank < 0 or rank >= self.world for rank in item["owners"])
                           for item in headers)
                    or any(item["expected_schema"] is not None and
                           item["expected_schema"] != (str(actual.dtype), tuple(actual.shape)) for item in headers)):
                raise ValueError(f"{name}: inconsistent check order, tensor schema, owners or duplicate case")

            flat = actual.detach().reshape(-1)
            reference = None if expected is None else expected.detach().reshape(-1)
            elements = max(1, self.chunk_bytes // actual.element_size())
            fingerprint = hashlib.sha256(repr(schema[1:3]).encode("utf-8"))
            candidate_mask = torch.zeros(self.world, dtype=torch.uint8, device="cpu")
            local_bad, local_nonfinite, first_bad = False, False, None
            peer_example = None
            for start in range(0, flat.numel(), elements):
                stop = min(start + elements, flat.numel())
                cpu = flat[start:stop].to("cpu").contiguous()
                value = cpu.view(torch.uint8)
                fingerprint.update(value.numpy().tobytes())
                if (cpu.is_floating_point() or cpu.is_complex()) and not torch.isfinite(cpu).all().item():
                    local_nonfinite = True
                if reference is not None:
                    correct = reference[start:stop].to("cpu").contiguous().view(torch.uint8)
                    if not torch.equal(value, correct):
                        local_bad = True
                        if first_bad is None:
                            offset = int(torch.nonzero(value != correct, as_tuple=False)[0, 0])
                            first_bad = start + offset // actual.element_size()
                else:
                    gathered = [torch.empty_like(value) for _ in range(self.world)] if self.rank == 0 else None
                    dist.gather(value, gather_list=gathered, dst=0, group=self.control)
                    if self.rank == 0:
                        classes = equivalence_classes(gathered)
                        if len(classes) > 1:
                            bad_ranks, attribution = candidate_ranks(classes)
                            candidate_mask[bad_ranks] = 1
                            if peer_example is None:
                                peer_example = {"equivalence_classes": classes,
                                                "attribution": attribution,
                                                "chunk_start_element": start}

            reports = [None] * self.world
            dist.all_gather_object(reports, {"bad": local_bad or local_nonfinite,
                                            "nonfinite": local_nonfinite, "first_bad_element": first_bad},
                                   group=self.control)
            if self.rank == 0:
                for rank, report in enumerate(reports):
                    if report["bad"]:
                        candidate_mask[list(headers[rank]["owners"])] = 1
                if candidate_mask.any().item():
                    candidates = torch.nonzero(candidate_mask, as_tuple=False).reshape(-1).tolist()
                    self._log("SDC_CANDIDATE", case=name, dtype=str(actual.dtype), shape=list(actual.shape),
                              comparison="known_reference" if reference is not None else "replica_bytes",
                              candidates=[self.devices[rank] for rank in candidates],
                              reports=[{"rank": rank, **report} for rank, report in enumerate(reports) if report["bad"]],
                              replica_difference=peer_example)
            dist.broadcast(candidate_mask, src=0, group=self.control)
            record_test(name, "FAIL" if candidate_mask.any().item() else "PASS",
                        dtype=str(actual.dtype), shape=list(actual.shape),
                        candidates=[self.devices[rank] for rank in range(self.world)
                                    if candidate_mask[rank].item()])
            self._candidates |= candidate_mask
            self._results[name + "/ok"] = torch.tensor(not candidate_mask.any().item(), dtype=torch.bool, device="cpu")
            self._results[name + "/sha256"] = torch.tensor(list(fingerprint.digest()), dtype=torch.uint8, device="cpu")
            self._count += 1

    def finish(self):
        if self._count < 1:
            raise ValueError("experiment did not execute any checks")
        self._results["checks/passed"] = torch.tensor(not self._candidates.any().item(), dtype=torch.bool, device="cpu")
        self._results["checks/candidates"] = self._candidates.clone()
        self._results["checks/count"] = torch.tensor(self._count, dtype=torch.int64, device="cpu")
        self._log("PASS" if self._results["checks/passed"].item() else "SDC_CANDIDATE_SUMMARY",
                  comparisons=self._count,
                  candidates=[self.devices[rank] for rank in range(self.world) if self._candidates[rank].item()])
        return self._results


def validate_results(results):
    """Offline-safe: no live process group, UUID lookup or communication."""
    if not isinstance(results, dict) or not results:
        raise ValidationError("expected a nonempty result dictionary")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in results.items()):
        raise ValidationError("results must be a flat dict[str, torch.Tensor]")
    passed, count, candidates = (results.get(key) for key in ("checks/passed", "checks/count", "checks/candidates"))
    if (passed is None or passed.dtype != torch.bool or passed.ndim != 0
            or count is None or count.dtype != torch.int64 or count.ndim != 0 or count.item() < 1
            or candidates is None or candidates.dtype != torch.uint8 or candidates.ndim != 1 or candidates.numel() < 2):
        raise ValidationError("missing or invalid screening metadata")
    cases = [key.removesuffix("/ok") for key in results if key.endswith("/ok")]
    expected_keys = {"checks/passed", "checks/count", "checks/candidates"} | {
        f"{case}/{field}" for case in cases for field in ("ok", "sha256")}
    if len(cases) != count.item() or set(results) != expected_keys:
        raise ValidationError("inconsistent screening cases or result keys")
    for case in cases:
        ok, digest = results[case + "/ok"], results[case + "/sha256"]
        if ok.dtype != torch.bool or ok.ndim != 0 or digest.dtype != torch.uint8 or tuple(digest.shape) != (32,):
            raise ValidationError(f"{case}: invalid result flag or fingerprint")
    if (not passed.item() or candidates.any().item()
            or any(not results[case + "/ok"].item() for case in cases)):
        raise ValidationError("SDC candidates detected; see report failures for GPU UUIDs and test cases")


class ScreeningExperiment(Experiment):
    """Common offline validation for experiments returning SDCChecks results."""

    def validate(self, results):
        super().validate(results)
        validate_results(results)


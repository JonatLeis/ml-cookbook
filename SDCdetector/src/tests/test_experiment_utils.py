"""Exercise the shared checker and real workloads on small CPU/Gloo replicas."""
from dataclasses import asdict, dataclass
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from configuration import load_config, load_experiment
from experiments import _utils as utils
from experiments import data_transfers, memory_patterns, transformer_tp
from reporting import Reporter
from validation import ValidationError, assert_exact


@dataclass
class CaseConfig:
    size: int = 11


def _checks_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=30))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    reporter = Reporter("zero")
    try:
        with reporter.activate():
            with utils.SDCChecks("cpu", CaseConfig(), 7, chunk_bytes=16) as checks:
                checks.tensor("equal", torch.arange(11, dtype=torch.float64))
                strided = torch.arange(12).reshape(3, 4)[:, ::2]
                checks.tensor("strided_reference", strided, strided.clone())
                healthy = checks.finish()
                utils.validate_results(healthy)
                assert healthy["checks/count"].item() == 2

            with utils.SDCChecks("cpu", CaseConfig(), 7, chunk_bytes=16) as checks:
                value = torch.arange(11, dtype=torch.float64)
                if rank == 2:
                    value[-1] += 1  # Also inspect the last partial chunk.
                checks.tensor("majority_tail", value)
                checks.tensor("ambiguous", torch.full((3,), float(rank)))
                value = torch.ones(3)
                if rank == 1:
                    value[1] = float("inf")
                checks.tensor("nonfinite", value)
                checks.tensor("signed_zero", torch.tensor([-0.0 if rank == 2 else 0.0]))
                checks.tensor("transfer", torch.tensor([rank == 1], dtype=torch.int64),
                              torch.zeros(1, dtype=torch.int64), owners=(rank, (rank - 1) % 3))
                checks.tensor("later_pass", torch.ones(3))
                failed = checks.finish()
                assert not failed["checks/passed"].item()
                try:
                    utils.validate_results(failed)
                    raise AssertionError("a later PASS erased earlier failures")
                except ValidationError:
                    pass

            with utils.SDCChecks("cpu", CaseConfig(), 7) as checks:
                def broken():
                    if rank == 1:
                        raise RuntimeError("local compute failed")
                    return torch.ones(1)
                try:
                    checks.call("operation", "compute", broken)
                    raise AssertionError("a rank-local error did not reach all peers")
                except RuntimeError as exc:
                    assert "local compute failed" in str(exc)
                # Schema failures must be coordinated before payload collectives.
                try:
                    checks.tensor("schema", torch.ones(rank + 1))
                    raise AssertionError("different tensor shapes were accepted")
                except ValueError:
                    pass
                checks.tensor("unique", torch.ones(1))
                try:
                    checks.tensor("unique", torch.ones(1))
                    raise AssertionError("duplicate case was accepted")
                except ValueError:
                    pass

            for config, seed in ((CaseConfig(rank), 7), (CaseConfig(), 7 + rank)):
                try:
                    with utils.SDCChecks("cpu", config, seed):
                        raise AssertionError("rank settings/seed disagreement was accepted")
                except ValueError:
                    pass

            # Shared input generation must not mutate the caller's RNG state.
            torch.manual_seed(123)
            before = torch.get_rng_state().clone()
            shared = utils.generate_shared_random_tensor((3, 5), "cpu", 7)
            assert torch.equal(before, torch.get_rng_state())
            assert_exact({"value": shared}, {"value": utils.gather_rank_tensors(shared)["rank_0000"]})

            # Full and quick experiments must execute identical work when given
            # identical overrides, including their offline validation schema.
            cases = {
                "elementwise": {"numel": 19, "iterations": 1, "chain_steps": 2,
                                "dtypes": ["float32", "bfloat16"], "integer_dtypes": ["int64"]},
                "indexing": {"shapes": [[3, 7]], "iterations": 1,
                             "dtypes": ["float32", "bfloat16", "int64"]},
                "local_reductions": {"rows": 3, "widths": [7], "iterations": 1,
                                     "dtypes": ["float32"]},
                "memory_patterns": {"memory_mib": 1, "chunk_mib": 1, "iterations": 1,
                                    "patterns": ["zeros", "index", "random"]},
            }
            for name, settings in cases.items():
                full = load_experiment(name, settings)
                quick = load_experiment(name + "_quick", settings)
                results = full.run("cpu", 7)
                full.validate(results)
                repeated = full.run("cpu", 7)
                quick_results = quick.run("cpu", 7)
                quick.validate(quick_results)
                assert_exact(results, repeated)
                assert_exact(results, quick_results)

        if rank == 0:
            Path(output).write_text(json.dumps({"report": reporter.snapshot(),
                                                "healthy": {key: value.tolist() for key, value in healthy.items()}}))
    finally:
        dist.destroy_process_group()


class ExperimentUtilsTests(unittest.TestCase):
    def test_seeded_generator_and_payload_markers(self):
        for seed in (0, 7, 2**63 - 1):
            a = torch.randn(11, generator=utils.cpu_generator(seed, 1))
            b = torch.randn(11, generator=utils.cpu_generator((seed + 1) % (2**63)))
            self.assertTrue(torch.equal(a, b))
        payload = data_transfers.Transfers._payload(64, 7, 2, 3, 4)
        self.assertEqual(payload[:24].tolist(), list(
            (2).to_bytes(8, "little") + (3).to_bytes(8, "little") + (4).to_bytes(8, "little")))
        self.assertTrue(torch.equal(payload, data_transfers.Transfers._payload(64, 7, 2, 3, 4)))
        self.assertFalse(torch.equal(payload, data_transfers.Transfers._payload(64, 7, 1, 3, 4)))

    def test_dtype_validation_and_byte_equivalence(self):
        utils.check_dtypes([torch.float16, torch.bfloat16])
        for values in ([], [torch.float32, torch.float32], [torch.int64]):
            with self.assertRaises(ValueError):
                utils.check_dtypes(values)
        values = [utils.tensor_bytes(torch.tensor([value])) for value in (0.0, 0.0, -0.0)]
        classes = utils.equivalence_classes(values)
        self.assertEqual(classes, [[0, 1], [2]])
        self.assertEqual(utils.candidate_ranks(classes), ([2], "majority_outlier"))
        self.assertEqual(utils.candidate_ranks([[0], [1]]), ([0, 1], "ambiguous_no_majority"))
        self.assertEqual(utils.equivalence_classes(values, (0, 2)), [[0], [2]])

    def test_device_identity_formats_and_tp_use_shared_helpers(self):
        raw = bytes.fromhex("00112233445566778899aabbccddeeff")
        with patch.object(utils.torch.cuda, "get_device_properties", return_value=SimpleNamespace(
                uuid=raw, name="test GPU")):
            info = utils.device_info(torch.device("cuda:0"), 2)
        self.assertEqual(info["uuid"], "GPU-00112233-4455-6677-8899-aabbccddeeff")
        self.assertEqual(info["rank"], 2)
        self.assertIsNone(utils.device_info(torch.device("cpu"), 0)["uuid"])
        self.assertIs(transformer_tp.equivalence_classes, utils.equivalence_classes)
        self.assertIs(transformer_tp.tensor_bytes, utils.tensor_bytes)

    def test_template_defaults_and_memory_entrypoint(self):
        entries = load_config(Path(__file__).resolve().parents[2] / "config.template.yml")["experiments"]
        for name, entry in entries.items():
            default = load_experiment(name, {})
            configured = load_experiment(name, entry["settings"])
            self.assertEqual(asdict(configured), asdict(default), name)
        self.assertIs(type(load_experiment("mem_pat", {})), memory_patterns.MemoryPatterns)

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo unavailable")
    def test_three_replicas_and_reused_workloads(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            mp.spawn(_checks_worker, args=((Path(directory) / "gloo").as_uri(), str(output)),
                     nprocs=3, join=True)
            data = json.loads(output.read_text())
        failures = {entry["case"]: entry for entry in data["report"]["failures"] if "case" in entry}
        for name, ranks in (("majority_tail", [2]), ("ambiguous", [0, 1, 2]), ("nonfinite", [1]),
                            ("signed_zero", [2]), ("transfer", [0, 1])):
            self.assertEqual([gpu["rank"] for gpu in failures[name]["candidates"]], ranks, name)
        self.assertEqual(failures["majority_tail"]["replica_difference"]["chunk_start_element"], 10)
        errors = [entry for entry in data["report"]["tests"] if entry["status"] == "ERROR"]
        self.assertEqual(errors[0]["details"]["errors"][0]["device"]["rank"], 1)
        self.assertTrue(data["healthy"]["checks/passed"])
        self.assertEqual(data["healthy"]["checks/count"], 2)

    def test_offline_schema_validation_rejects_missing_evidence(self):
        for value in ({}, {"checks/passed": torch.tensor(True)},
                      {"checks/passed": torch.tensor(True), "checks/count": torch.tensor(1),
                       "checks/candidates": torch.zeros(3, dtype=torch.uint8)}):
            with self.assertRaises(ValidationError):
                utils.validate_results(value)

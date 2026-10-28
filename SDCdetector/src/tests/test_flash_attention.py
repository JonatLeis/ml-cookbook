"""Independent FA dispatch and replica checks; no installed FA/CUDA required."""
from contextlib import contextmanager
from dataclasses import replace
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

from configuration import describe, load_experiment
from experiments import _flash_attention as shared
from experiments import fa2, fa3, fa4
from experiments._utils import SDCChecks
from reporting import Reporter
from validation import ValidationError


MODULES = (fa2, fa3, fa4)


def _replica_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=30))
    torch.use_deterministic_algorithms(True)
    reporter = Reporter("zero")
    try:
        calls, active, patched_backwards = [], [], []

        @contextmanager
        def stat_context():
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        def load(module):
            calls.append(module.EXPERIMENT.name)
            version = int(module.EXPERIMENT.name[-1])

            def backend(q, k, v, **options):
                assert not active  # Forward is never patched.
                assert (options.get("deterministic") is True) == experiment.backward
                assert type(options["causal"]) is bool
                assert ("dropout_p" in options) == (version == 2)
                if version == 2:
                    assert options["dropout_p"] == 0.0
                # Different generations deliberately produce different results.
                result = (q + k + v) * version
                if experiment.backward:
                    def check_scope(grad):
                        patched = version == 4 and experiment.workaround
                        assert bool(active) == patched
                        if patched:
                            patched_backwards.append(True)
                    result.register_hook(check_scope)
                return (result, None) if version == 4 else result

            return backend

        common = dict(shapes=((1, 4, 1, 8),), dtypes=[torch.float16],
                      causal_modes=(False,), data_patterns=("uniform",))
        forwards = []
        with patch.object(shared, "SDCChecks", side_effect=lambda device, config, seed:
                          SDCChecks("cpu", config, seed, chunk_bytes=16)), reporter.activate():
            for module in MODULES:
                experiment = replace(module.EXPERIMENT, **common)
                with patch.object(module, "_load_backend", side_effect=lambda: load(module)), \
                        patch.object(fa4, "stat_loads", return_value=stat_context) as configure:
                    results = experiment.run("cuda", seed=7)
                    experiment.validate(results)
                    if module is fa4:
                        configure.assert_called_once()
                    else:
                        configure.assert_not_called()
                forwards.append(next(tensor for key, tensor in results.items()
                                     if key.endswith("/forward/sha256")))
            assert calls == ["fa2", "fa3", "fa4"]
            assert len(patched_backwards) == 1
            assert not torch.equal(forwards[0], forwards[1])

            # A rank-local FA3 backend error remains ERROR; FA4 can run afterward.
            experiment = fa3.FA3(**dict(common, shapes=((1, 1, 1, 256),)))
            def unsupported(q, k, v, **options):
                assert options["deterministic"] is True
                if rank == 1:
                    raise RuntimeError("Deterministic backward not supported for hdim 256.")
                return q + k + v
            with patch.object(fa3, "_load_backend", return_value=unsupported):
                try:
                    experiment.run("cuda", seed=7)
                    raise AssertionError("unsupported FA3 kernel was accepted")
                except RuntimeError as exc:
                    assert "fa3/" in str(exc) and "hdim 256" in str(exc)

            # Both forward-only and workaround=false bypass the SM90 helper.
            for backward in (False, True):
                experiment = fa4.FA4(**dict(common, backward=backward), workaround=False)
                with patch.object(fa4, "_load_backend", side_effect=lambda: load(fa4)), \
                        patch.object(fa4, "stat_loads") as configure:
                    results = experiment.run("cuda", seed=7)
                    experiment.validate(results)
                    configure.assert_not_called()
                assert any("/dq/" in key for key in results) == backward
            experiment = fa4.FA4(**dict(common, backward=False, iterations=2,
                                      dtypes=[torch.float16, torch.bfloat16],
                                      causal_modes=(False, True),
                                      data_patterns=("normal", "uniform", "zeros")))
            with patch.object(fa4, "_load_backend", side_effect=lambda: load(fa4)), \
                    patch.object(fa4, "stat_loads") as configure:
                experiment.validate(experiment.run("cuda", seed=7))
                configure.assert_not_called()  # Even workaround=true is ignored without backward.

            experiment = fa4.FA4(**common)
            def configure_stat_loads(device):
                if rank == 1:
                    raise RuntimeError("incompatible SM90 helper")
                return stat_context
            with patch.object(fa4, "_load_backend", side_effect=lambda: load(fa4)), \
                    patch.object(fa4, "stat_loads", side_effect=configure_stat_loads):
                try:
                    experiment.run("cuda", seed=7)
                    raise AssertionError("rank-local setup error was not propagated")
                except RuntimeError as exc:
                    assert "fa4/backward_setup" in str(exc) and "incompatible SM90 helper" in str(exc)

            with SDCChecks("cpu", experiment, 7, chunk_bytes=16) as checks:
                def broken():
                    if rank == 1:
                        raise RuntimeError("unsupported deterministic kernel")
                try:
                    checks.call("fa4", "compute", broken)
                    raise AssertionError("rank-local compute error was not propagated")
                except RuntimeError as exc:
                    assert "unsupported deterministic kernel" in str(exc)
                actual = torch.arange(8, dtype=torch.float32)
                if rank == 2:
                    actual[3] += 1
                checks.tensor("majority", actual)
                checks.tensor("ambiguous", torch.full((8,), float(rank)))
                actual = torch.ones(8)
                if rank == 1:
                    actual[2] = float("nan")
                checks.tensor("nonfinite", actual)
                try:
                    experiment.validate(checks.finish())
                    raise AssertionError("candidate results unexpectedly passed validation")
                except ValidationError:
                    pass
        if rank == 0:
            Path(output).write_text(json.dumps(reporter.snapshot()))
    finally:
        dist.destroy_process_group()


class FlashAttentionTests(unittest.TestCase):
    def tearDown(self):
        for module in MODULES:
            module._load_backend.cache_clear()

    def test_independent_lazy_backend_loading(self):
        for module in MODULES:
            module._load_backend.cache_clear()
            function = object()
            with patch.object(fa2, "package_version", return_value="2.8.0"), patch.object(
                    module, "import_module", return_value=SimpleNamespace(flash_attn_func=function)) as importer:
                self.assertIs(module.EXPERIMENT._load_backend(), function)
                self.assertIs(module.EXPERIMENT._load_backend(), function)
                importer.assert_called_once_with(module.EXPERIMENT.backend)
        fa2._load_backend.cache_clear()
        with patch.object(fa2, "package_version", return_value="1.0.9"), self.assertRaises(RuntimeError):
            fa2.EXPERIMENT._load_backend()

    def test_options_tuple_returns_and_autograd(self):
        q = torch.ones(1, 2, 1, 8, requires_grad=True)
        for module in MODULES:
            experiment = module.EXPERIMENT
            calls = []
            def backend(q, k, v, **options):
                calls.append(options)
                return q + k + v, None
            output = shared._call_backend(backend, q, q, q, experiment.name, experiment._options(True))
            self.assertEqual(torch.autograd.grad(output.sum(), q)[0].tolist(), torch.full_like(q, 3).tolist())
            self.assertIs(calls[0]["deterministic"], True)
            self.assertEqual("dropout_p" in calls[0], module is fa2)
            self.assertNotIn("deterministic", replace(experiment, backward=False)._options(False))
        with self.assertRaises(ValueError):
            shared._call_backend(lambda *args, **kwargs: torch.ones(3), q, q, q, "fa4", {})
        with self.assertRaises(TypeError):
            shared._call_backend(lambda *args, **kwargs: None, q, q, q, "fa3", {})
        with self.assertRaises(TypeError):
            shared._call_backend(lambda q, k, v, causal: q, q, q, q, "fa4", fa4.EXPERIMENT._options(False))

    def test_patterns_seed_and_independent_config(self):
        for module in MODULES:
            config = load_experiment(module.EXPERIMENT.name, {"shapes": [[1, 8, 2, 64]],
                                                            "dtypes": ["bfloat16"], "backward": False})
            config._check_config()
            self.assertEqual(config.shapes, ((1, 8, 2, 64),))
            fields = describe(module.EXPERIMENT.name)
            self.assertNotIn("fa_versions", fields)
            self.assertNotIn("fa4_stat_loads", fields)
            self.assertEqual("workaround" in fields, module is fa4)
            for invalid in ({"fa_versions": [2]}, {"fa4_stat_loads": "auto"}):
                with self.assertRaises(ValueError):
                    load_experiment(module.EXPERIMENT.name, invalid)
            if module is not fa4:
                with self.assertRaises(ValueError):
                    load_experiment(module.EXPERIMENT.name, {"workaround": False})
        for pattern in ("normal", "uniform", "zeros"):
            a = shared._cpu_data((1, 8, 2, 64), torch.bfloat16, pattern, .25, torch.Generator().manual_seed(7))
            b = shared._cpu_data((1, 8, 2, 64), torch.bfloat16, pattern, .25, torch.Generator().manual_seed(7))
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.isfinite(a).all())
        for settings in ({"shapes": ((1, 2, 3),)}, {"shapes": ((1, 0, 2, 8),)},
                         {"dtypes": [torch.float32]}, {"causal_modes": ()}, {"data_patterns": ("unknown",)},
                         {"input_scale": float("nan")}, {"iterations": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(config, **settings)._check_config()
        for value in ("false", 0, None):
            with self.assertRaises(ValueError):
                load_experiment("fa4", {"workaround": value})
            with self.assertRaises(ValueError):
                fa4.FA4(workaround=value)._check_config()

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is not available")
    def test_three_cpu_replicas_and_failure_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "gloo").as_uri()
            output = str(Path(directory) / "report.json")
            mp.spawn(_replica_worker, args=(rendezvous, output), nprocs=3, join=True)
            data = json.loads(Path(output).read_text())
        failures = {event.get("case"): event for event in data["failures"] if event.get("case")}
        self.assertEqual([gpu["rank"] for gpu in failures["majority"]["candidates"]], [2])
        self.assertEqual(failures["majority"]["replica_difference"]["attribution"], "majority_outlier")
        self.assertEqual([gpu["rank"] for gpu in failures["ambiguous"]["candidates"]], [0, 1, 2])
        self.assertEqual(failures["ambiguous"]["replica_difference"]["attribution"], "ambiguous_no_majority")
        self.assertEqual([gpu["rank"] for gpu in failures["nonfinite"]["candidates"]], [1])
        errors = [test for test in data["tests"] if test["status"] == "ERROR"]
        self.assertEqual({test["name"] for test in errors}, {
            "fa4/backward_setup", "fa4/compute", "fa3/float16/1x1x1x256/uniform/iteration_000/causal_0/compute"})
        self.assertTrue(all(test["details"]["errors"][0]["device"]["rank"] == 1 for test in errors))

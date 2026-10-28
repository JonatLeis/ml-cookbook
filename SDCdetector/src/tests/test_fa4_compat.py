"""FA4 workaround/cache lifecycle checks without importing CUDA extensions."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from configuration import describe, load_config, load_experiment
from experiments import fa4_compat as compat
from experiments import fa4


class FA4CompatibilityTests(unittest.TestCase):
    def tearDown(self):
        compat._sm90_stat_loads.cache_clear()

    def test_boolean_switch_and_sm90_guard_with_original_path_on_blackwell(self):
        context = object()
        with patch.object(compat, "_sm90_stat_loads", return_value=SimpleNamespace(applied=context)) as make, \
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)) as capability:
            self.assertIsNone(fa4.FA4(workaround=False)._backward_context(torch.device("cuda:0")))
            capability.assert_not_called()
            make.assert_not_called()
            self.assertIs(fa4.FA4(workaround=True)._backward_context(torch.device("cuda:0")), context)
            capability.return_value = (10, 0)
            capability.reset_mock()
            self.assertIsNone(fa4.FA4(workaround=False)._backward_context(torch.device("cuda:0")))
            capability.assert_not_called()
            with self.assertRaisesRegex(ValueError, "requires an SM90"):
                fa4.FA4(workaround=True)._backward_context(torch.device("cuda:0"))
            with self.assertRaisesRegex(ValueError, "requires an SM90"):
                compat.stat_loads(torch.device("cpu"))
            self.assertEqual(make.call_count, 1)

    def test_cache_isolation_reuse_and_restoration_including_errors(self):
        original_copy = lambda: None
        replacement = object()
        class PersistentCache(dict):
            def clear(self):
                raise AssertionError("must not clear a potentially persistent library cache")

        original_cache = PersistentCache({"same-shape": "unpatched kernel"})
        backward = SimpleNamespace(compile_cache=original_cache)
        owner = SimpleNamespace(cpasync_bulk_get_copy_fn=original_copy)
        modules = {"flash_attn.cute.flash_bwd_sm90": SimpleNamespace(copy_utils=owner),
                   "flash_attn.cute.interface": SimpleNamespace(_flash_attn_bwd=backward)}
        with patch.object(compat, "import_module", side_effect=modules.__getitem__), \
                patch.object(compat, "_make_stat_copy", return_value=replacement):
            probe = compat._sm90_stat_loads()
            self.assertIs(probe, compat._sm90_stat_loads())
        with probe.applied():
            self.assertIs(owner.cpasync_bulk_get_copy_fn, replacement)
            self.assertEqual(backward.compile_cache, {})
            probe.stats["compile_calls"] += 4
            backward.compile_cache["same-shape"] = "patched kernel"
        self.assertIs(backward.compile_cache, original_cache)
        self.assertIs(owner.cpasync_bulk_get_copy_fn, original_copy)
        with probe.applied():
            self.assertEqual(backward.compile_cache["same-shape"], "patched kernel")
        self.assertEqual(original_cache, {"same-shape": "unpatched kernel"})

        # A new cache entry must also compile the replacement, even if an older
        # shape already passed. Reject an incompatible kernel using another path.
        with self.assertRaisesRegex(RuntimeError, "not compiled"):
            with probe.applied():
                backward.compile_cache["unsupported-shape"] = "unpatched kernel"
        self.assertEqual(probe.cache, {})
        with self.assertRaisesRegex(RuntimeError, "not compiled"):
            with probe.applied():
                pass
        with self.assertRaisesRegex(RuntimeError, "backward failed"):
            with probe.applied():
                probe.stats["compile_calls"] += 1
                backward.compile_cache["partial"] = "broken kernel"
                raise RuntimeError("backward failed")
        self.assertEqual(probe.cache, {})
        self.assertIs(backward.compile_cache, original_cache)
        self.assertIs(owner.cpasync_bulk_get_copy_fn, original_copy)
        self.assertEqual(original_cache, {"same-shape": "unpatched kernel"})

    def test_independent_yaml_presets_and_fa4_flag_without_fa_imports(self):
        expected = ((1, 1, 1, 64), (1, 128, 2, 64), (1, 129, 2, 64), (2, 256, 4, 64),
                    (2, 512, 4, 64), (1, 257, 2, 96), (2, 256, 4, 128), (1, 128, 2, 256))
        with patch.object(compat, "import_module", side_effect=AssertionError("unexpected FA4 import")):
            for filename in ("config.yml", "config.template.yml"):
                entries = load_config(Path(__file__).resolve().parents[2] / filename)["experiments"]
                self.assertNotIn("flash_attention", entries)
                for name in ("fa2", "fa3", "fa4"):
                    shapes = expected[:-1] if name == "fa3" else expected
                    defaults = describe(name)
                    self.assertEqual(defaults["shapes"]["default"], [list(shape) for shape in shapes])
                    self.assertEqual("workaround" in defaults, name == "fa4")
                    experiment = load_experiment(name, entries[name]["settings"])
                    experiment._check_config()
                    self.assertEqual(experiment.shapes, shapes)
                    self.assertEqual(experiment.dtypes, [torch.float16, torch.bfloat16])
                    if name == "fa4":
                        self.assertIs(experiment.workaround, True)

"""Small end-to-end CPU checks; no synthetic faults inside experiments."""
import json
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiment import Experiment
from reporting import Reporter
from validation import ValidationError
import run as runner
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def invoke(self, name, output, *extra):
        return subprocess.run([sys.executable, str(ROOT / "run.py"), name,
                               "--output", str(output), *map(str, extra)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_transformer_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("transformer",):
                with self.subTest(experiment=name):
                    output = Path(directory) / name
                    result = self.invoke(name, output)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads((output / "report.json").read_text())
                    self.assertEqual(report["status"], "PASS")
                    self.assertEqual(report["completed_repeats"], 2)

    def test_saved_reference_and_mismatched_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("transformer",):
                reference = root / (name + "-reference")
                result = self.invoke(name, reference, "--save-tensors")
                self.assertEqual(result.returncode, 0, result.stderr)
                result = self.invoke(name, root / (name + "-verify"), "--reference", reference)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = self.invoke("transformer", root / "mismatch", "--reference", root / "transformer-reference",
                                 "--seed", "7")
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((root / "mismatch" / "report.json").read_text())
            self.assertEqual(report["status"], "ERROR")
            self.assertIn("metadata differs", report["error"])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "keep.txt"
            marker.write_text("keep")
            result = self.invoke("transformer", root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(marker.read_text(), "keep")
            self.assertFalse((root / "results.pt").exists())


@dataclass
class SmallExperiment(Experiment):
    fail: bool = False

    def run(self, device, seed):
        if self.fail:
            raise ValidationError("sum: flat element 3 differs")
        return {"sum": torch.arange(4, device=device)}


class ModeTests(unittest.TestCase):
    def test_success_and_failure_in_all_modes(self):
        for mode in ("zero", "minimum", "verbose"):
            for fail in (False, True):
                with self.subTest(mode=mode, fail=fail), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "run"
                    reporter = Reporter(mode)
                    with patch.object(runner, "importlib") as modules:
                        modules.import_module.return_value = SimpleNamespace(EXPERIMENT=SmallExperiment(fail))
                        code = runner.main(["small", "--output", str(output)], reporter)
                    self.assertEqual(code, int(fail))
                    data = reporter.snapshot()
                    self.assertEqual(data["status"], "FAIL" if fail else "PASS")
                    self.assertEqual(data["metadata"]["seed"], 12345)
                    self.assertEqual(data["metadata"]["config"], {"fail": fail})
                    if fail:
                        self.assertIn("flat element 3", data["error"])
                        self.assertEqual(data["tests"][-1]["repetition"], 1)
                        self.assertEqual(data["tests"][-1]["name"], "execution")
                    if mode == "zero":
                        self.assertFalse(output.exists())
                    else:
                        files = {path.name for path in output.iterdir()}
                        if mode == "minimum":
                            self.assertEqual(files, {"report.json", "run.log"})
                            log = (output / "run.log").read_text()
                            self.assertIn(f"Running Experiment small with setting: fail={str(fail).lower()}", log)
                            self.assertIn(f"Experiment small {'FAILED' if fail else 'PASSED'}!", log)
                            self.assertEqual(log.count("Experiment summary"), 1)
                            if fail:
                                self.assertIn("flat element 3 differs", log)
                        elif not fail:
                            self.assertTrue({"results.pt", "latest.pt", "metadata.json"} <= files)

    def test_setup_errors_are_recorded_in_minimum_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            reporter = Reporter("minimum")
            with patch.object(runner, "importlib") as modules:
                modules.import_module.side_effect = ImportError("missing experiment")
                self.assertEqual(runner.main(["missing", "--output", str(output)], reporter), 1)
            data = json.loads((output / "report.json").read_text())
            self.assertEqual(data["status"], "ERROR")
            self.assertIn("missing experiment", data["error"])
            log = (output / "run.log").read_text()
            self.assertIn("Running Experiment missing with setting:", log)
            self.assertIn("Experiment missing FAILED!", log)
            self.assertIn("ImportError: missing experiment", log)


    def test_zero_requires_no_output(self):
        reporter = Reporter("zero")
        with patch.object(runner, "importlib") as modules:
            modules.import_module.return_value = SimpleNamespace(EXPERIMENT=SmallExperiment())
            self.assertEqual(runner.main(["small"], reporter), 0)


if __name__ == "__main__":
    unittest.main()

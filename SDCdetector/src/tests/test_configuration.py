"""Config validation uses dataclasses and CPU tensors; no GPU work."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from configuration import load_config, load_experiment, describe
from controller import main


class ConfigurationTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text(text)
            return load_config(path)

    def test_nested_shapes_dtypes_and_defaults(self):
        experiment = load_experiment("indexing_quick", {"shapes": [[3, 5]], "dtypes": ["int64"]})
        self.assertEqual(experiment.shapes, ((3, 5),))
        self.assertEqual(experiment.dtypes, [torch.int64])
        self.assertNotEqual(load_experiment("indexing_quick", {}).shapes, ((3, 5),))
        self.assertIn("shapes", describe("indexing_quick"))
        self.assertNotIn("shapes", describe("elementwise_quick"))

    def test_invalid_fields_and_types(self):
        for name, settings in (("elementwise_quick", {"shapes": [[2, 3]]}),
                               ("indexing_quick", {"shapes": [[2, 3, 4]]}),
                               ("indexing_quick", {"dtypes": ["not_a_dtype"]}),
                               ("transformer", {"steps": True}),
                               ("transformer", {"learning_rate": float("nan")})):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                load_experiment(name, settings)

    def test_yaml_errors(self):
        for text in ("version: 1\ncontroller: {loging: zero}\nexperiments: {}",
                     "version: 1\ncontroller: {repeats: true}\nexperiments: {}",
                     "version: 1\nexperiments: {transformer: {enabled: yesplease}}",
                     "version: 1\nexperiments: {transformer: {run: {threads: 0}}}",
                     "version: 1\nversion: 1\nexperiments: {}", "[invalid"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_repository_config_and_dry_check(self):
        path = Path(__file__).resolve().parents[2] / "config.yml"
        data = load_config(path)
        for name, entry in data["experiments"].items():
            load_experiment(name, entry["settings"])
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--config", str(path), "--check-config"]), 0)
        self.assertIn("no experiments executed", output.getvalue())

    def test_cpu_yaml_overrides_reach_worker_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text('''version: 1
controller:
  device: cpu
  nproc_per_node: 1
  logging: minimum
  output: output
  api_port: 0
  api_linger: 0
  repeats: 2
experiments:
  transformer:
    run: {repeats: 1, seed: 7}
    environment: {CONFIG_TEST_MARKER: yes}
    settings: {steps: 1, width: 8, heads: 2, length: 4, vocab_size: 8}
''')
            result = subprocess.run([sys.executable, "controller.py", "--config", str(config)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((root / "output" / "report.json").read_text())
            rank = data["experiments"]["transformer"]["ranks"]["0"]
            self.assertEqual(rank["metadata"]["config"]["steps"], 1)
            self.assertEqual(rank["metadata"]["config"]["width"], 8)
            self.assertEqual(rank["metadata"]["seed"], 7)
            self.assertEqual(rank["requested_repeats"], 1)
            log = (root / "output" / "run.log").read_text()
            self.assertIn("steps=1", log)
            self.assertIn("width=8", log)
            self.assertIn("batch_size=2", log)  # Resolved default, not an explicit override.
            self.assertIn("seed=7, repeats=1", log)

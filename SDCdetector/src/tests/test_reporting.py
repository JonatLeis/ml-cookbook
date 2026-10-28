"""Reporting and HTTP checks require only the Python standard library."""
import json
from pathlib import Path
import tempfile
import unittest
from urllib.request import urlopen

from reporting import Reporter, diagnostic, progress, record_test


class ReportingTests(unittest.TestCase):
    def test_modes_have_identical_results_and_different_logs(self):
        for mode in ("zero", "minimum", "verbose"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                reporter = Reporter(mode)
                reporter.attach(directory)
                reporter.experiment_log.start("example", {"dtype": "float32", "shape": [8]})
                with reporter.activate():
                    with progress("addition"):
                        record_test("addition", "PASS")
                    diagnostic("MISMATCH", case="cast", first_differing_element=7,
                               candidates=[{"rank": 1, "uuid": "GPU-example"}])
                    record_test("cast", "FAIL", dtype="float32", shape=[8])
                reporter.finish("FAIL", "cast differs")
                reporter.experiment_log.finish("example", "FAIL", "cast differs")
                reporter.experiment_log.summary()
                reporter.write(directory)
                data = reporter.snapshot()
                self.assertEqual([t["status"] for t in data["tests"]], ["PASS", "FAIL"])
                self.assertEqual(data["failures"][0]["first_differing_element"], 7)
                self.assertIn("finished_at", data)
                if mode == "zero":
                    self.assertEqual(list(Path(directory).iterdir()), [])
                elif mode == "minimum":
                    log = (Path(directory) / "run.log").read_text()
                    self.assertIn('Running Experiment example with setting: dtype="float32", shape=[8]...', log)
                    self.assertIn("Experiment example FAILED!", log)
                    self.assertIn("Reason: cast differs", log)
                    self.assertNotIn("CHECK_START", log)
                    self.assertFalse((Path(directory) / "events.jsonl").exists())
                    self.assertEqual(json.loads((Path(directory) / "report.json").read_text()), data)
                else:
                    lines = [json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines()]
                    self.assertEqual(any(e["event"] == "trace" for e in lines), mode == "verbose")
                    self.assertEqual(json.loads((Path(directory) / "report.json").read_text()), data)

    def test_minimum_summary_is_grouped_stably_and_keeps_full_settings_above(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter = Reporter("minimum")
            reporter.attach(directory)
            shapes = [[1, length, 4, 64] for length in range(32)]
            for name, status in (("bad_first", "FAIL"), ("good_first", "PASS"),
                                 ("bad_second", "ERROR"), ("good_second", "PASS")):
                reporter.experiment_log.start(name, {"shapes": shapes, "backward": True})
                reporter.experiment_log.finish(name, status)
            reporter.experiment_log.summary()
            log = (Path(directory) / "run.log").read_text()
            self.assertIn('shapes=' + json.dumps(shapes, separators=(",", ":")) + ', backward=true', log)
            rows = [line for line in log.splitlines() if " | PASSED" in line or " | FAILED" in line]
            self.assertEqual([line.split(" (")[0] for line in rows],
                             ["good_first", "good_second", "bad_first", "bad_second"])
            self.assertTrue(all("...)" in row for row in rows))
            self.assertTrue(all(len(row) < 160 for row in rows))

    def test_empty_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            reporter = Reporter("minimum")
            reporter.attach(directory)
            reporter.experiment_log.summary()
            self.assertIn("Experiment summary", (Path(directory) / "run.log").read_text())

    def test_http_and_prometheus(self):
        reporter = Reporter("zero")
        reporter.test('a"b\\c\nd', "PASS")
        reporter.repetition = 2
        reporter.test('a"b\\c\nd', "FAIL", error="mismatch")
        reporter.finish("FAIL")
        with reporter.serve() as (host, port):
            with urlopen(f"http://{host}:{port}/results") as response:
                self.assertEqual(json.load(response), reporter.snapshot())
            with urlopen(f"http://{host}:{port}/metrics") as response:
                self.assertIn("version=0.0.4", response.headers["Content-Type"])
                metrics = response.read().decode()
        self.assertIn("tiny_experiments_run_status 2", metrics)
        self.assertIn('tiny_experiments_test_status{test="a\\"b\\\\c\\nd"} 2', metrics)
        self.assertEqual(metrics.count('tiny_experiments_test_status{'), 1)
        self.assertIn("tiny_experiments_test_timestamp_seconds{", metrics)
        self.assertNotIn("mismatch", metrics)

    def test_snapshot_is_independent(self):
        reporter = Reporter()
        data = reporter.snapshot()
        data["tests"].append({})
        self.assertEqual(reporter.snapshot()["tests"], [])


if __name__ == "__main__":
    unittest.main()

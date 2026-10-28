"""Controller tests use small subprocesses and local IPC, never GPU workloads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.request import urlopen

from controller import Controller


ROOT = Path(__file__).resolve().parents[1]
FAKE_WORKER = '''
import json, os, socket, sys
from reporting import Reporter, compact
name, status = sys.argv[1:]
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
    channel.connect(os.environ["EXPERIMENT_CONTROLLER_SOCKET"])
    with channel.makefile("w") as stream:
        def send(message):
            stream.write(compact(message) + "\\n")
            stream.flush()
        send({"type": "hello", "experiment": name, "rank": 0})
        reporter = Reporter(os.environ["EXPERIMENT_LOGGING"], sink=send)
        reporter.update(metadata={"experiment": name, "seed": 7, "rank": 0})
        reporter.event("trace", debug=True, message="trace detail")
        reporter.test("addition", status, error="element 2 differs", shape=[4])
        if status == "FAIL":
            reporter.diagnostic("MISMATCH", case="addition", first_differing_element=2)
        reporter.finish(status)
sys.exit(status != "PASS")
'''


class ControllerTests(unittest.TestCase):
    def fake(self, name, status="PASS"):
        return [sys.executable, "-c", FAKE_WORKER, name, status]

    def test_sequential_jobs_retained_at_one_endpoint_in_all_modes(self):
        for mode in ("zero", "minimum", "verbose"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "suite"
                controller = Controller(["first", "second"], mode=mode, output=output)
                with controller.serve() as (host, port):
                    def command(name):
                        if name == "second":
                            with urlopen(f"http://{host}:{port}/results") as response:
                                data = json.load(response)
                            self.assertEqual(data["experiments"]["first"]["status"], "PASS")
                            self.assertEqual(data["experiments"]["second"]["status"], "RUNNING")
                        return self.fake(name)
                    self.assertEqual(controller.run_suite(command), 0)
                    with urlopen(f"http://{host}:{port}/metrics") as response:
                        metrics = response.read().decode()
                    for name in ("first", "second"):
                        self.assertIn(f'tiny_experiments_run_status{{experiment="{name}",rank="0"}} 1', metrics)
                    self.assertEqual(metrics.count("# TYPE tiny_experiments_run_status gauge"), 1)
                if mode == "zero":
                    self.assertFalse(output.exists())
                elif mode == "minimum":
                    self.assertEqual({p.name for p in output.iterdir()}, {"report.json", "run.log"})
                    log = (output / "run.log").read_text()
                    self.assertIn("Running Experiment first with setting: ranks=1...", log)
                    self.assertIn("Experiment second PASSED!", log)
                    self.assertNotIn("trace detail", log)
                    self.assertEqual(log.count("Experiment summary"), 1)
                else:
                    self.assertEqual({p.name for p in output.iterdir()}, {"report.json", "events.jsonl"})
                    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
                    traces = [event for event in events if event["event"] == "worker"
                              and event["entry"]["event"] == "trace"]
                    self.assertEqual(bool(traces), mode == "verbose")
                    self.assertEqual(json.loads((output / "report.json").read_text())["status"], "PASS")

    def test_minimum_log_is_live_and_summarizes_failed_and_interrupted_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suite"
            controller = Controller(["bad", "good", "cancelled", "pending"], mode="minimum", output=output)
            options = {name: {"settings": {"dtypes": ["float16"], "shapes": [[1, 128, 2, 64]]},
                              "seed": 7, "repeats": 2} for name in ("bad", "good", "cancelled")}
            def command(name):
                log = (output / "run.log").read_text()
                self.assertIn(f"Running Experiment {name} with setting:", log)
                if name == "good":
                    self.assertIn("Experiment bad FAILED!", log)
                if name == "cancelled":
                    raise KeyboardInterrupt("test cancellation")
                return self.fake(name, "FAIL" if name == "bad" else "PASS")
            with self.assertRaises(KeyboardInterrupt):
                controller.run_suite(command, job_options=options)
            log = (output / "run.log").read_text()
            self.assertIn('dtypes=["float16"], shapes=[[1,128,2,64]], seed=7, repeats=2, ranks=1', log)
            self.assertIn("Reason: KeyboardInterrupt: test cancellation", log)
            self.assertIn("Reason: rank 0:", log)
            self.assertIn("element 2 differs", log)
            rows = [line for line in log.splitlines() if " | PASSED" in line or " | FAILED" in line]
            self.assertEqual([line.split(" (")[0] for line in rows], ["good", "bad", "cancelled"])
            self.assertNotIn("pending", log)
            self.assertEqual(log.count("Experiment cancelled FAILED!"), 1)
            self.assertEqual(controller.snapshot()["experiments"]["pending"]["status"], "PENDING")

    def test_failed_job_preserves_context_and_next_job_runs(self):
        controller = Controller(["bad", "good"], mode="zero")
        self.assertEqual(controller.run_suite(lambda name: self.fake(name, "FAIL" if name == "bad" else "PASS")), 1)
        data = controller.snapshot()
        self.assertEqual(data["status"], "FAIL")
        self.assertEqual(data["experiments"]["good"]["status"], "PASS")
        failed = data["experiments"]["bad"]["ranks"]["0"]
        self.assertEqual(failed["failures"][0]["first_differing_element"], 2)
        self.assertEqual(failed["metadata"]["seed"], 7)
        self.assertIn("timestamp", failed["tests"][0])

    def test_missing_rank_or_failed_launcher_is_error(self):
        controller = Controller(["missing"], ranks=2, mode="zero")
        self.assertEqual(controller.run_suite(lambda name: self.fake(name)), 1)
        self.assertEqual(controller.snapshot()["experiments"]["missing"]["ranks"]["1"]["status"], "ERROR")
        controller = Controller(["launcher"], mode="zero")
        command = [sys.executable, "-c", 'print("x" * 20000); raise SystemExit(9)']
        self.assertEqual(controller.run_suite(lambda name: command), 1)
        data = controller.snapshot()["experiments"]["launcher"]
        self.assertEqual(data["exit_code"], 9)
        self.assertLessEqual(len(data["launcher_tail"]), 8192)

    def test_timeout_terminates_job(self):
        controller = Controller(["slow"], mode="zero")
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        self.assertEqual(controller.run_suite(lambda name: command, timeout=.1), 1)
        self.assertIn("TimeoutExpired", controller.snapshot()["experiments"]["slow"]["error"])
        self.assertIsNone(controller.process)

    def test_interrupted_suite_marks_active_job_and_retains_previous_results(self):
        controller = Controller(["first", "interrupted"], mode="zero")
        def command(name):
            if name == "interrupted":
                raise KeyboardInterrupt("test cancellation")
            return self.fake(name)
        with self.assertRaises(KeyboardInterrupt):
            controller.run_suite(command)
        data = controller.snapshot()
        self.assertEqual(data["status"], "ERROR")
        self.assertEqual(data["experiments"]["first"]["status"], "PASS")
        self.assertEqual(data["experiments"]["interrupted"]["status"], "ERROR")
        data["experiments"]["first"]["command"].append("changed")
        self.assertNotIn("changed", controller.snapshot()["experiments"]["first"]["command"])

    def test_actual_cpu_worker_in_all_modes(self):
        # Verify the real worker/runner boundary with the existing small CPU experiment.
        for mode in ("zero", "minimum", "verbose"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "suite"
                result = subprocess.run([
                    sys.executable, str(ROOT / "controller.py"), "transformer", "--device", "cpu",
                    "--nproc-per-node", "1", "--repeats", "1", "--api-port", "0",
                    "--api-linger", "0", "--logging", mode, "--output", str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if mode == "zero":
                    self.assertFalse(output.exists())
                else:
                    report = json.loads((output / "report.json").read_text())
                    self.assertEqual(report["experiments"]["transformer"]["ranks"]["0"]["status"], "PASS")
                    if mode == "minimum":
                        self.assertEqual({p.name for p in output.iterdir()}, {"report.json", "run.log"})
                        log = (output / "run.log").read_text()
                        self.assertIn("steps=5, batch_size=2, length=16", log)
                        self.assertIn("seed=12345, repeats=1", log)
                    else:
                        self.assertTrue((output / "transformer" / "results.pt").exists())

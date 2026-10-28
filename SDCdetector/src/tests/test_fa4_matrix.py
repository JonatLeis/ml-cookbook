"""CPU-only matrix planning/supervision tests; no FA4 or CUDA imports."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[2] / "src_test" / "fa_matrix.py"
SPEC = importlib.util.spec_from_file_location("fa4_matrix", PATH)
matrix = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(PATH.parent), *sys.path]):
    SPEC.loader.exec_module(matrix)


def records(stream):
    prefix = matrix.debugger.PREFIX
    return [json.loads(line[len(prefix):]) for line in stream.getvalue().splitlines()
            if line.startswith(prefix)]


class FA4MatrixTests(unittest.TestCase):
    def plan(self, options=()):
        with patch.object(matrix, "run_matrix", return_value=0) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(matrix.main(list(options)), 0)
        return run.call_args.args

    def test_default_matrix_covers_dtype_shape_mask_and_backward_settings(self):
        args, tests = self.plan()
        self.assertEqual(len(tests), 64)
        self.assertEqual(len({matrix.test_details(test)["test_id"] for test in tests}), 64)
        self.assertEqual({t.dtype for t in tests}, {"float16", "bfloat16"})
        self.assertEqual({tuple(t.shape) for t in tests}, set(matrix.DEFAULT_SHAPES))
        self.assertEqual({t.causal for t in tests}, {False, True})
        self.assertEqual({t.case for t in tests}, {"backward-false", "backward-true"})
        self.assertTrue(all(t.strict_torch == (t.case == "backward-true") for t in tests))
        self.assertTrue(all(t.direct_stat_loads and t.check_reference and t.repeats == 2 for t in tests))
        self.assertTrue(all(t.reference_policy == "eager" and not t.single_thread_backward for t in tests))

    def test_custom_matrix_and_reproduction_command_preserve_settings(self):
        _, tests = self.plan(["--dtypes", "float16", "--shape", "1", "129", "2", "64",
                              "--shape", "2", "256", "4", "128", "--causal-modes", "on",
                              "--cases", "forward", "backward-true", "--seeds", "3", "5",
                              "--repeats", "3", "--timeout", "45", "--gradient", "float-sum"])
        self.assertEqual(len(tests), 8)
        for test in tests:
            command = matrix.debugger.child_command(test, test.case)
            self.assertEqual(command[command.index("--seed") + 1], str(test.seed))
            self.assertEqual(command[command.index("--repeats") + 1], "3")
            self.assertIn("--causal", command)
            self.assertEqual("--direct-stat-loads" in command, test.case != "forward")
            self.assertEqual("--strict-torch" in command, test.case == "backward-true")
            self.assertNotIn("--single-thread-backward", command)
            reproduce = matrix.reproduction_command(test)
            self.assertNotIn("--worker", reproduce)
            self.assertIn("--timeout 45.0", reproduce)
            self.assertIn("--reference-policy eager", reproduce)
        _, controls = self.plan(["--original-stat-loads"])
        self.assertTrue(all(not t.direct_stat_loads for t in controls))

    def test_list_works_without_site_packages_or_gpu_imports(self):
        result = subprocess.run([sys.executable, "-S", str(PATH), "--list"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = [json.loads(line[len(matrix.debugger.PREFIX):]) for line in result.stdout.splitlines()]
        self.assertEqual(len(events), 65)
        self.assertEqual(events[-1]["planned"], 64)
        self.assertTrue(all(e["event"] == "PLAN" for e in events))

    def test_errors_and_timeouts_continue_and_are_never_reported_as_pass(self):
        args, tests = self.plan(["--shape", "1", "128", "2", "64", "--causal-modes", "off"])
        for bad_status, expected_exit in (("ERROR", 1), ("TIMEOUT", 124)):
            results = [{"case": "test", "status": bad_status if i == 0 else "PASS",
                        "process_exited": True, "exit_code": 1 if i == 0 else 0}
                       for i in range(len(tests))]
            stream = io.StringIO()
            with patch.object(matrix.debugger, "run_process", side_effect=results) as run, redirect_stdout(stream):
                self.assertEqual(matrix.run_matrix(args, tests), expected_exit)
            self.assertEqual(run.call_count, 4)
            self.assertTrue(all(call.kwargs == {"verbose": False} for call in run.call_args_list))
            self.assertEqual(len({tuple(call.args[0]) for call in run.call_args_list}), 4)
            summary = records(stream)[-1]
            self.assertEqual(summary["status"], bad_status)
            self.assertEqual(summary["counts"][bad_status], 1)
            self.assertEqual(summary["not_run"], 0)
            failed = summary["failures"][0]
            self.assertEqual(failed["shape"], [1, 128, 2, 64])
            self.assertIn("--timeout", failed["reproduce"])

    def test_unreaped_worker_and_interrupt_leave_an_incomplete_summary(self):
        args, tests = self.plan(["--shape", "1", "128", "2", "64", "--causal-modes", "off"])
        for result, status, exit_code in (({"status": "TIMEOUT", "process_exited": False}, "TIMEOUT", 124),
                                           (KeyboardInterrupt(), "INTERRUPTED", 130)):
            stream = io.StringIO()
            effect = result if isinstance(result, BaseException) else None
            with patch.object(matrix.debugger, "run_process", return_value=result, side_effect=effect) as run, \
                    redirect_stdout(stream):
                self.assertEqual(matrix.run_matrix(args, tests), exit_code)
            self.assertEqual(run.call_count, 1)
            summary = records(stream)[-1]
            self.assertEqual(summary["status"], status)
            self.assertGreater(summary["not_run"], 0)

    def test_complete_success_and_verbose_forwarding(self):
        args, tests = self.plan(["--dtypes", "float16", "--shape", "1", "128", "2", "64",
                                 "--cases", "backward-true", "--causal-modes", "off", "--verbose"])
        stream = io.StringIO()
        with patch.object(matrix.debugger, "run_process", return_value={"status": "PASS", "process_exited": True}) as run, \
                redirect_stdout(stream):
            self.assertEqual(matrix.run_matrix(args, tests), 0)
        self.assertEqual(run.call_args.kwargs, {"verbose": True})
        summary = records(stream)[-1]
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["counts"]["PASS"], 1)
        self.assertFalse(summary["failures"])
        self.assertEqual(summary["not_run"], 0)

    def test_compact_supervisor_keeps_failure_details_without_streaming_worker(self):
        failed = {"phase": "repeat-1/reference_check", "event": "INFO", "matches": False,
                  "tensor": "dv", "policy": "eager", "max_abs_diff": .1}
        error = {"phase": "case_result", "event": "ERROR", "error": "reference failed"}
        code = ("print('worker noise'); "
                f"print({(matrix.debugger.PREFIX + json.dumps(failed))!r}); "
                f"print({(matrix.debugger.PREFIX + json.dumps(error))!r}); raise SystemExit(1)")
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = matrix.debugger.run_process([sys.executable, "-c", code], "example", 5, verbose=False)
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error"], "reference failed")
        self.assertEqual(result["failed_checks"][0]["tensor"], "dv")

    def test_invalid_or_duplicate_matrix_settings_are_rejected(self):
        for options in (["--shape", "1", "0", "2", "64"], ["--timeout", "nan"],
                        ["--repeats", "0"], ["--seeds", "-1"], ["--stack-interval", "-1"],
                        ["--dtypes", "float16", "float16"], ["--cases", "forward", "forward"],
                        ["--seeds", "0", "0"],
                        ["--shape", "1", "2", "3", "4", "--shape", "1", "2", "3", "4"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                matrix.main(options)

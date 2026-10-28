"""Check the FA4 debugger with CPU mock kernels and short child processes."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.fa4_compat import _make_stat_copy


PATH = Path(__file__).resolve().parents[2] / "src_test" / "fa_test.py"
SPEC = importlib.util.spec_from_file_location("fa4_debugger", PATH)
debugger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(debugger)


def records(stream):
    return [json.loads(line[len(debugger.PREFIX):]) for line in stream.getvalue().splitlines()
            if line.startswith(debugger.PREFIX)]


class FA4DebuggerTests(unittest.TestCase):
    def args(self, case="backward-true", gradient="upstream"):
        return SimpleNamespace(case=case, gradient=gradient, repeats=2, causal=False,
                               shape=[1, 4, 2, 8], dtype="bfloat16", seed=7, strict_torch=False,
                               device="cuda:0", stack_interval=0, single_thread_backward=False,
                               scalar_dq_store=False, direct_stat_loads=False, check_reference=False,
                               reference_policy="elementwise")

    def test_eager_policy_cli_is_explicit_and_enables_reference(self):
        with patch.object(debugger, "run_process", return_value={"status": "PASS", "process_exited": True}) as run, \
                redirect_stdout(io.StringIO()):
            debugger.main(["--case", "forward", "--reference-policy", "eager"])
        command = run.call_args.args[0]
        self.assertIn("--check-reference", command)
        self.assertEqual(command[command.index("--reference-policy") + 1], "eager")
        with patch.object(debugger, "run_process", return_value={"status": "PASS", "process_exited": True}) as run, \
                redirect_stdout(io.StringIO()):
            debugger.main(["--case", "forward", "--check-reference"])
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--reference-policy") + 1], "elementwise")

    def test_eager_budget_accepts_rounding_but_rejects_large_or_widespread_errors(self):
        expected = torch.zeros(1024, dtype=torch.float64)
        eager = expected.clone()
        eager[0] = 1e-4
        actual = expected.clone()
        actual[0] = 8e-5
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertTrue(debugger.check_reference(torch, (actual,), (expected,), "forward", "reference",
                                                     "bfloat16", baseline=(eager,)))
        event = records(stream)[0]
        self.assertFalse(event["elementwise_matches"])
        self.assertTrue(event["matches"])
        self.assertEqual(event["allowed_max_abs_diff"], 2e-4)
        for corrupted in (torch.full_like(actual, 1), torch.full_like(actual, 8e-5)):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "mismatch"):
                debugger.check_reference(torch, (corrupted,), (expected,), "forward", "reference",
                                         "bfloat16", baseline=(eager,))
        for invalid in (torch.full_like(eager, float("nan")), torch.zeros(1)):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "invalid"):
                debugger.check_reference(torch, (actual,), (expected,), "forward", "reference",
                                         "bfloat16", baseline=(invalid,))

    def test_eager_reference_output_and_gradients_for_dtype_mask_and_gradient_modes(self):
        for dtype in ("bfloat16", "float16"):
            for causal in (False, True):
                for gradient in ("upstream", "float-sum"):
                    args = self.args("backward-false", gradient)
                    args.dtype, args.causal = dtype, causal
                    generator = torch.Generator().manual_seed(args.seed)
                    inputs = [(torch.randn(args.shape, generator=generator) * .25).to(getattr(torch, dtype))
                              for _ in range(3)]
                    upstream = (torch.randn(args.shape, generator=generator) * .125).to(getattr(torch, dtype))
                    reference = debugger.cpu_reference(torch, inputs, upstream, args)
                    eager = debugger.cpu_reference(torch, inputs, upstream, args, low_precision=True)
                    self.assertEqual(len(eager), 4)
                    self.assertTrue(all(t.dtype == getattr(torch, dtype) and torch.isfinite(t).all() for t in eager))
                    with redirect_stdout(io.StringIO()):
                        self.assertTrue(debugger.check_reference(torch, eager, reference, args.case, "reference",
                                                                 dtype, baseline=eager))

    def test_stat_probe_cli_requires_backward_and_enables_reference(self):
        for case in debugger.CASES[1:]:
            with patch.object(debugger, "run_process", return_value={"status": "PASS",
                                                                   "process_exited": True}) as run, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(debugger.main(["--case", case, "--direct-stat-loads"]), 0)
            self.assertIn("--direct-stat-loads", run.call_args.args[0])
            self.assertIn("--check-reference", run.call_args.args[0])
        for case in ("all", "forward"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                debugger.main(["--case", case, "--direct-stat-loads"])

    def test_stat_probe_restores_helper_and_accepts_compiled_kernel_reuse(self):
        original, replacement = object(), object()
        owner = SimpleNamespace(cpasync_bulk_get_copy_fn=original)
        stats = {"compile_calls": 0}
        with patch.object(debugger, "import_module", return_value=SimpleNamespace(copy_utils=owner)), \
                patch.object(debugger, "make_direct_stat_copy", return_value=replacement), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "not compiled"):
                with debugger.stat_load_probe("backward-false", True, stats):
                    self.assertIs(owner.cpasync_bulk_get_copy_fn, replacement)
            self.assertIs(owner.cpasync_bulk_get_copy_fn, original)
            with debugger.stat_load_probe("backward-false", True, stats):
                stats["compile_calls"] += 1
            with debugger.stat_load_probe("backward-false", True, stats):
                pass  # The second repetition may use the cached patched kernel.
            with self.assertRaisesRegex(RuntimeError, "backward failed"):
                with debugger.stat_load_probe("backward-false", True, stats):
                    raise RuntimeError("backward failed")
        self.assertIs(owner.cpasync_bulk_get_copy_fn, original)

    def test_backward_context_excludes_forward_and_reference(self):
        active, entries = [], []

        @contextmanager
        def probe():
            entries.append(True)
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        def backend(q, k, v, **options):
            self.assertFalse(active)
            output = q + k + v
            output.register_hook(lambda grad: self.assertTrue(active))
            return output, None

        with redirect_stdout(io.StringIO()):
            debugger.exercise(backend, torch, torch.device("cpu"), self.args(), lambda: None, probe)
        self.assertEqual(len(entries), 2)
        self.assertFalse(active)

    def test_direct_stat_load_copies_whole_tile_from_one_lane(self):
        class Value:
            def __init__(self, value):
                self.value = value

            def ir_value(self, **kwargs):
                return self.value

        class Pointer:
            def __init__(self, name):
                self.name = name

            def toint(self, **kwargs):
                return Value(self.name)

        fp32 = SimpleNamespace(width=32)

        class Tensor:
            shape = (128, 2)
            element_type = fp32

            def __init__(self, memspace):
                self.memspace = memspace

            def __getitem__(self, index):
                return SimpleNamespace(iterator=Pointer((self.memspace, index)))

        elected, calls = [], []

        @contextmanager
        def elect_one():
            elected.append(True)
            try:
                yield
            finally:
                elected.pop()

        def inline_asm(*args, **kwargs):
            self.assertTrue(elected)
            calls.append((args, kwargs))

        cute = SimpleNamespace(Pointer=Pointer, AddressSpace=SimpleNamespace(gmem="g", smem="s"),
                               group_modes=lambda t, *_: t, rank=lambda t: len(t.shape),
                               size=lambda shape: math.prod(shape),
                               arch=SimpleNamespace(elect_one=elect_one))
        llvm = SimpleNamespace(inline_asm=inline_asm, AsmDialect=SimpleNamespace(AD_ATT="att"))
        modules = {"cutlass": SimpleNamespace(cute=cute, Float32=fp32, Int32=Value),
                   "cutlass.cute": cute,
                   "cutlass.cutlass_dsl": SimpleNamespace(dsl_user_op=lambda fn: fn),
                   "cutlass._mlir": SimpleNamespace(dialects=SimpleNamespace(llvm=llvm)),
                   "cutlass._mlir.dialects": SimpleNamespace(llvm=llvm)}
        # Exercise both the standalone validated probe and its production copy.
        for make in (lambda stats: debugger.make_direct_stat_copy("backward-false", stats), _make_stat_copy):
            with self.subTest(factory=make), patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
                self._check_stat_copy(make, Tensor, Pointer, calls)

    def _check_stat_copy(self, make, Tensor, Pointer, calls):
        calls.clear()
        Tensor.shape = (128, 2)
        stats = {"compile_calls": 0}
        factory = make(stats)
        copy = factory(Tensor("g"), Tensor("s"))
        copy(src_idx=0, dst_idx=1, tma_bar_ptr=Pointer("barrier"))
        self.assertEqual(stats["compile_calls"], 1)
        self.assertEqual(calls[0][0][1], [("g", (None, 0)), ("s", (None, 1)), "barrier", 512])
        self.assertIn("mbarrier::complete_tx::bytes", calls[0][0][2])
        with self.assertRaises(ValueError):
            factory(Tensor("s"), Tensor("g"))
        Tensor.shape = (3, 2)  # A 12-byte bulk transfer is invalid.
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            factory(Tensor("g"), Tensor("s"))(0, 0, Pointer("barrier"))

    def test_reference_agrees_with_sdpa_output_and_gradients(self):
        def backend(q, k, v, **options):
            output = torch.nn.functional.scaled_dot_product_attention(
                *[tensor.transpose(1, 2).float() for tensor in (q, k, v)],
                is_causal=options["causal"])
            return output.transpose(1, 2).to(q.dtype), None

        for case in ("forward", "backward-false"):
            for gradient in ("upstream", "float-sum"):
                for causal in (False, True):
                    for dtype in ("float16", "bfloat16"):
                        with self.subTest(case=case, gradient=gradient, causal=causal, dtype=dtype):
                            args = self.args(case, gradient)
                            args.check_reference, args.causal, args.dtype = True, causal, dtype
                            for policy in ("elementwise", "eager"):
                                args.reference_policy = policy
                                output = io.StringIO()
                                with redirect_stdout(output):
                                    debugger.exercise(backend, torch, torch.device("cpu"), args, lambda: None)
                                checked = [event for event in records(output) if "matches" in event
                                           and event["phase"].endswith("/reference_check")]
                                self.assertEqual(len(checked), args.repeats * (1 if case == "forward" else 4))
                                self.assertTrue(all(event["matches"] for event in checked))
                                self.assertTrue(all(event["policy"] == policy for event in checked))

    def test_reference_rejects_finite_but_incorrect_gradient(self):
        expected = tuple(torch.ones(3, dtype=torch.float64) for _ in range(4))
        actual = list(expected)
        actual[1] = actual[1] + 1
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "mismatch for dq"):
            debugger.check_reference(torch, actual, expected, "backward-false", "reference", "bfloat16")
        checked = records(output)
        self.assertEqual([r["tensor"] for r in checked], ["output", "dq", "dk", "dv"])
        self.assertEqual(checked[1]["mismatched_elements"], 3)
        self.assertEqual(checked[1]["actual_at_worst"], 2)
        self.assertEqual(checked[1]["reference_at_worst"], 1)
        self.assertGreater(checked[1]["max_tolerance_ratio"], 1)

    def test_output_mismatch_still_reports_all_gradients_and_repetitions(self):
        def backend(q, k, v, **options):
            output = torch.nn.functional.scaled_dot_product_attention(
                *[t.transpose(1, 2).float() for t in (q, k, v)])
            return output.transpose(1, 2).to(q.dtype) + 1, None

        args, stream = self.args("backward-false"), io.StringIO()
        args.check_reference = True
        with redirect_stdout(stream), self.assertRaisesRegex(ValueError, "repetitions \\[1, 2\\]"):
            debugger.exercise(backend, torch, torch.device("cpu"), args, lambda: None)
        events = records(stream)
        before = [r for r in events if r["phase"].endswith("/reference_forward") and "matches" in r]
        checked = [r for r in events if r["phase"].endswith("/reference_check") and "matches" in r]
        self.assertEqual(len(before), 2)
        self.assertTrue(all(not r["matches"] for r in before))
        self.assertEqual([r["tensor"] for r in checked], ["output", "dq", "dk", "dv"] * 2)
        self.assertTrue(all(r["matches"] for r in checked if r["tensor"] != "output"))
        self.assertTrue(all(r["unchanged"] for r in events if "unchanged" in r))
        self.assertFalse(any(r["event"] == "PASS" for r in events))

    def test_backward_output_mutation_is_detected(self):
        def backend(q, k, v, **options):
            output = q + k + v

            def corrupt(gradient):
                output.detach().add_(1)
                return gradient

            output.register_hook(corrupt)
            return output, None

        args, stream = self.args("backward-false"), io.StringIO()
        args.check_reference = True
        with redirect_stdout(stream), self.assertRaisesRegex(ValueError, "integrity"):
            debugger.exercise(backend, torch, torch.device("cpu"), args, lambda: None)
        integrity = [r for r in records(stream) if "unchanged" in r]
        self.assertEqual(len(integrity), 2)
        self.assertTrue(all(not r["unchanged"] and r["max_abs_change"] > 0 for r in integrity))

    def test_scalar_probe_patches_actual_dependency_and_restores_on_error(self):
        original, replacement = object(), object()
        owner = SimpleNamespace(cpasync_reduce_bulk_add_f32=original, __file__="/quack/copy_utils.py")
        backend = SimpleNamespace(copy_utils=owner)
        with patch.object(debugger, "import_module", return_value=backend), \
                patch.object(debugger, "make_scalar_reduce", return_value=replacement), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                with debugger.dq_store_probe("backward-false", True):
                    self.assertIs(owner.cpasync_reduce_bulk_add_f32, replacement)
                    raise RuntimeError("kernel failure")
        self.assertIs(owner.cpasync_reduce_bulk_add_f32, original)

    def test_scalar_probe_rejects_unused_replacement(self):
        original = object()
        owner = SimpleNamespace(cpasync_reduce_bulk_add_f32=original)
        with patch.object(debugger, "import_module", return_value=SimpleNamespace(copy_utils=owner)), \
                patch.object(debugger, "make_scalar_reduce", return_value=object()), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "not compiled"):
            with debugger.dq_store_probe("backward-false", True):
                pass
        self.assertIs(owner.cpasync_reduce_bulk_add_f32, original)

    def test_scalar_probe_cli_requires_nondeterministic_case_and_enables_reference(self):
        with patch.object(debugger, "run_process", return_value={"case": "backward-false", "status": "PASS",
                                                               "process_exited": True}) as run, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(debugger.main(["--case", "backward-false", "--scalar-dq-store"]), 0)
        command = run.call_args.args[0]
        self.assertIn("--scalar-dq-store", command)
        self.assertIn("--check-reference", command)
        for options in ([], ["--case", "backward-true"],
                        ["--case", "backward-false", "--strict-torch"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                debugger.main([*options, "--scalar-dq-store"])

    def test_single_thread_backward_scopes_autograd_setting(self):
        for single_thread in (False, True):
            for gradient in ("upstream", "float-sum"):
                args = self.args(gradient=gradient)
                args.single_thread_backward = single_thread
                observed = []

                def backend(q, k, v, **options):
                    output = q + k + v
                    output.register_hook(lambda grad: observed.append(
                        torch.autograd.is_multithreading_enabled()))
                    return output, None

                with torch.autograd.set_multithreading_enabled(True):
                    with redirect_stdout(io.StringIO()):
                        debugger.exercise(backend, torch, torch.device("cpu"), args, lambda: None)
                    self.assertTrue(torch.autograd.is_multithreading_enabled())
                self.assertEqual(observed, [not single_thread] * args.repeats)

    def test_loaded_runtime_reports_unique_actual_mappings(self):
        mappings = ("1-2 r-xp 0 00:00 1 /runtime/libcute_dsl_runtime.so\n"
                    "2-3 rw-p 0 00:00 1 /runtime/libcute_dsl_runtime.so\n"
                    "3-4 r-xp 0 00:00 2 /runtime/_cutlass_ir.so\n"
                    "4-5 r-xp 0 00:00 3 /other/libc.so\n")
        output = io.StringIO()
        with patch.object(debugger.Path, "read_text", return_value=mappings), redirect_stdout(output):
            debugger.loaded_runtime("backward-false")
        self.assertEqual(records(output)[0]["libraries"],
                         ["/runtime/_cutlass_ir.so", "/runtime/libcute_dsl_runtime.so"])
        output = io.StringIO()
        with patch.object(debugger.Path, "read_text", side_effect=OSError("no procfs")), redirect_stdout(output):
            debugger.loaded_runtime("forward")
        self.assertEqual(records(output)[0]["libraries"], [])

    def test_tuple_autograd_modes_and_repeated_inputs(self):
        for case in debugger.CASES:
            for gradient in ("upstream", "float-sum"):
                with self.subTest(case=case, gradient=gradient):
                    args, calls = self.args(case, gradient), []
                    def backend(q, k, v, **options):
                        calls.append((options, q.detach().clone(), q.requires_grad))
                        return q + k + v, torch.ones(1)
                    output = io.StringIO()
                    with redirect_stdout(output):
                        debugger.exercise(backend, torch, torch.device("cpu"), args, lambda: None)
                    self.assertTrue(torch.equal(calls[0][1], calls[1][1]))
                    self.assertEqual(calls[0][2], case != "forward")
                    self.assertEqual(calls[0][0].get("deterministic"),
                                     {"backward-true": True, "backward-false": False}.get(case))
                    events = records(output)
                    self.assertEqual(events[-1]["event"], "PASS")
                    self.assertTrue(any(event["phase"].endswith("/unwrap") and event["event"] == "END"
                                        for event in events))
                    self.assertEqual(any(event["phase"].endswith("/backward_call") for event in events),
                                     case != "forward")

    def test_error_is_attached_to_actual_phase(self):
        output = io.StringIO()
        def unsupported(*args, **kwargs):
            raise RuntimeError("unsupported deterministic kernel")
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "unsupported"):
            debugger.exercise(unsupported, torch, torch.device("cpu"), self.args(), lambda: None)
        last = records(output)[-1]
        self.assertEqual(last["phase"], "repeat-1/forward_call")
        self.assertEqual(last["event"], "ERROR")

    def test_nonfinite_output_is_not_a_pass(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "NaN"):
            debugger.exercise(lambda q, k, v, **kwargs: q * float("nan"), torch, torch.device("cpu"),
                              self.args(case="forward"), lambda: None)
        self.assertEqual(records(output)[-1]["phase"], "repeat-1/finite_checks")

    def test_supervisor_timeout_reports_last_phase(self):
        command = [sys.executable, "-u", "-c", 'import time; print(\'FA4_DEBUG {"phase":"backward_call","event":"BEGIN"}\', flush=True); time.sleep(30)']
        with redirect_stdout(io.StringIO()):
            result = debugger.run_process(command, "backward-true", .5)
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["last_worker_phase"], "backward_call")
        self.assertTrue(result["process_exited"])

    def test_supervisor_child_exit_codes(self):
        for code in (0, 3):
            with redirect_stdout(io.StringIO()):
                result = debugger.run_process([sys.executable, "-c", f"raise SystemExit({code})"], "forward", 5)
            self.assertEqual(result["status"], "PASS" if code == 0 else "ERROR")
            self.assertEqual(result["exit_code"], code)

    def test_default_cases_use_fresh_processes_and_continue_after_timeout(self):
        commands = []
        def run(command, case, timeout):
            commands.append(command)
            return {"case": case, "status": "TIMEOUT" if case == "backward-default" else "PASS",
                    "process_exited": True}
        with patch.object(debugger, "run_process", side_effect=run), redirect_stdout(io.StringIO()):
            self.assertEqual(debugger.main(["--stack-interval", "0"]), 124)
        self.assertEqual(len(commands), 4)
        self.assertEqual([command[command.index("--case") + 1] for command in commands], list(debugger.CASES))
        self.assertTrue(all("--worker" in command for command in commands))
        commands.clear()
        with patch.object(debugger, "run_process", side_effect=run), redirect_stdout(io.StringIO()):
            debugger.main(["--case", "backward-false", "--single-thread-backward"])
        self.assertIn("--single-thread-backward", commands[0])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            debugger.main(["--timeout", "0"])

"""Run one experiment, validate and compare independent repetitions"""
import argparse
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
from datetime import timedelta
import socket
import sys
import time
from contextlib import nullcontext

from reporting import Reporter, compact, debug
from configuration import configure_experiment
import shutil

# Must be set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist

from experiment import Experiment
from validation import ValidationError, assert_exact, assert_finite



def configure(seed, threads):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)


def main(argv=None, reporter=None):
    """Run the CLI or supply a Reporter to collect results without any files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="module name in experiments/, e.g. transformer")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, help="new output directory (required unless logging is zero)")
    parser.add_argument("--logging", choices=("zero", "minimum", "verbose"), default="minimum")
    parser.add_argument("--save-tensors", action="store_true", help="retain tensors for offline reference comparisons")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, help="HTTP port; each distributed rank adds its rank")
    parser.add_argument("--api-linger", type=float, default=0, help="seconds to keep completed results available for scraping")
    parser.add_argument("--settings-json", default="{}", help="experiment dataclass overrides (JSON object)")
    parser.add_argument("--reference", type=Path, help="previous successful output directory")
    args = parser.parse_args(argv)
    if reporter is not None:
        args.logging = reporter.mode
    if args.logging != "zero" and args.output is None:
        parser.error("--output is required for minimum/verbose logging")
    if args.logging == "zero" and reporter is None and args.api_port is None:
        parser.error("zero logging requires --api-port or a supplied Reporter")
    if args.logging == "zero" and args.save_tensors:
        parser.error("--save-tensors requires minimum/verbose logging")
    if args.api_linger < 0 or not math.isfinite(args.api_linger):
        parser.error("--api-linger must be finite and nonnegative")
    if args.api_port is not None and not 0 <= args.api_port + int(os.environ.get("RANK", 0)) <= 65535:
        parser.error("API port plus rank must be in [0, 65535]")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.experiment):
        parser.error("experiment must be a simple module name")
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    reporter = reporter or Reporter(args.logging)
    reporter.update(metadata={"experiment": args.experiment, "seed": args.seed,
                              "device": args.device, "threads": args.threads,
                              "rank": int(os.environ.get("RANK", 0))},
                    requested_repeats=args.repeats,
                    reference=str(args.reference.resolve()) if args.reference else None)
    server = (reporter.serve(args.api_host, args.api_port + int(os.environ.get("RANK", 0)))
              if args.api_port is not None else nullcontext())
    with reporter.activate(), server:
        try:
            if "RANK" in os.environ:
                suffix = f"rank-{int(os.environ['RANK']):04d}"
                if args.output is not None:
                    args.output = args.output / suffix
                if args.reference is not None:
                    args.reference = args.reference / suffix
            if args.logging != "zero" and (not reporter.managed or args.logging == "verbose" or args.save_tensors):
                # Reserve output before setup so initialization errors are recorded too.
                args.output.mkdir(parents=True, exist_ok=False)
                args.output_created = True
                if not reporter.managed:
                    reporter.attach(args.output)
            reporter.event("run_started", metadata=reporter.snapshot()["metadata"])
            return execute(args, parser, reporter)
        except Exception as exc:
            status = "FAIL" if isinstance(exc, ValidationError) else "ERROR"
            reporter.test("runner", status, error=f"{type(exc).__name__}: {exc}")
            reporter.finish(status, f"{type(exc).__name__}: {exc}")
            if args.logging != "zero" and not getattr(args, "output_created", False):
                # No writable run directory exists for this setup failure.
                print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        finally:
            if reporter.snapshot()["status"] == "RUNNING":
                reporter.finish("ERROR", "run interrupted during setup or execution")
            # Setup can fail before the effective experiment configuration is known.
            reporter.experiment_log.start(args.experiment, {
                "settings": args.settings_json, "seed": args.seed, "repeats": args.repeats,
                "threads": args.threads, "device": args.device})
            data = reporter.snapshot()
            reporter.experiment_log.finish(args.experiment, data["status"], data.get("error"))
            reporter.experiment_log.summary()
            if not reporter.managed and args.logging != "zero" and getattr(args, "output_created", False):
                reporter.write(args.output)
            if args.api_port is not None and args.api_linger:
                time.sleep(args.api_linger)
            if dist.is_initialized():
                dist.destroy_process_group()


def execute(args, parser, reporter):
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.device == "cuda" and os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        parser.error("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    module = importlib.import_module(f"experiments.{args.experiment}")
    experiment = module.EXPERIMENT
    if not isinstance(experiment, Experiment):
        parser.error("module.EXPERIMENT must be an Experiment instance")

    experiment = configure_experiment(experiment, json.loads(args.settings_json))
    reporter.experiment_log.start(args.experiment, {
        **asdict(experiment), "seed": args.seed, "repeats": args.repeats,
        "threads": args.threads, "device": args.device})

    # Initialize distributed environment if RANK variable is set. Usually done by torchrun or torch.distributed.launch. If not set, run in single-process mode.
    distributed = "RANK" in os.environ

    if distributed:
        if args.device != "cuda":
            parser.error("Distributed execution currently requires --device cuda")

        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank >= torch.cuda.device_count():
            parser.error("LOCAL_RANK exceeds the number of visible GPUs")

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        debug(f"Initializing process group: host={socket.gethostname()} rank={os.environ['RANK']}")
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=120),
            device_id=torch.device(f"cuda:{local_rank}")
        )

        rank = dist.get_rank()
        debug(f"Process group ready: rank={rank}/{dist.get_world_size()} device={device}")
    else:
        device = torch.device(args.device)

    configure(args.seed, args.threads)
    metadata = {
        "format_version": 1,
        "experiment": args.experiment,
        "config": json.loads(json.dumps(asdict(experiment), default=str)),
        "seed": args.seed,
        "device": args.device,
        "threads": args.threads,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "rank": dist.get_rank() if distributed else 0,
        "world_size": dist.get_world_size() if distributed else 1,
        "host": socket.gethostname(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unknown")) if device.type == "cuda" else None,
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("NCCL_", "CUDA_", "TORCH_"))},
    }
    reporter.update(metadata=metadata)
    save_tensors = args.logging != "zero" and (args.save_tensors or args.logging == "verbose")
    if save_tensors:
        (args.output / "metadata.json").write_text(compact(metadata) + "\n")
    reporter.event("configuration", metadata=metadata)
    phase = "reference"
    repetition = 0
    reference = None
    try:
        if args.reference:
            old_report = json.loads((args.reference / "report.json").read_text())
            old_metadata = json.loads((args.reference / "metadata.json").read_text())
            if old_report["status"] != "PASS":
                raise ValueError("reference run did not pass")
            # Placement may change between independent jobs.
            compatibility = {key: value for key, value in metadata.items()
                             if key not in ("host", "gpu_uuid", "environment")}
            if any(old_metadata.get(key) != value for key, value in compatibility.items()):
                differences = sorted(key for key, value in compatibility.items()
                                     if old_metadata.get(key) != value)
                raise ValueError(f"reference metadata differs: {differences}")
            reference = torch.load(args.reference / "results.pt", map_location="cpu", weights_only=True)
            experiment.validate(reference)
        first = None
        for repetition in range(1, args.repeats + 1):
            reporter.repetition = repetition
            phase = "execution"
            debug(f"Starting repetition {repetition}/{args.repeats}")
            configure(args.seed, args.threads)
            results = experiment.run(device, args.seed)
            # Contract checking before conversion produces clear errors for invalid results.
            phase = "finite_check"
            try:
                assert_finite(results)
            except ValidationError:
                # Preserve tensor results, including NaN/Inf, for diagnosis.
                if save_tensors and isinstance(results, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                     for k, v in results.items()):
                    torch.save(results, args.output / f"failed-repeat-{repetition}.pt")
                raise
            results = {key: tensor.detach().cpu().clone() for key, tensor in results.items()}
            filename = "results.pt" if repetition == 1 else "latest.pt"
            if save_tensors:
                torch.save(results, args.output / filename)
            phase = "experiment_validation"
            experiment.validate(results)
            phase = "reference_comparison"
            if reference is not None:
                assert_exact(reference, results)
            phase = "repeat_comparison"
            if first is not None:
                assert_exact(first, results)
            else:
                first = results
            reporter.test("repetition", "PASS")
            reporter.update(completed_repeats=repetition)
        phase = "synchronization"
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if distributed:
            dist.barrier()
        if args.logging == "verbose" and (not distributed or dist.get_rank() == 0):
            snapshot_dir = args.output / "inspector-prom-snapshot"
            for source in sorted(Path("/dumps").glob("*.prom")):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, snapshot_dir / source.name)
                debug(f"Inspector snapshot: {source.name}")
        reporter.finish("PASS")
    except Exception as exc:
        status = "FAIL" if isinstance(exc, ValidationError) else "ERROR"
        reporter.test(phase, status, error=f"{type(exc).__name__}: {exc}")
        reporter.finish(status, f"{type(exc).__name__}: {exc}")
        if args.logging == "verbose":
            import traceback
            debug(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

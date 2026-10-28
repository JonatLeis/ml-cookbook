"""Runs isolated experiment jobs sequentially behind one stable results endpoint"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time

from reporting import Reporter, timestamp, compact
from configuration import load_config, load_experiment, describe, validate_run


STATUS_VALUE = {"PENDING": -1, "RUNNING": 0, "PASS": 1, "FAIL": 2, "ERROR": 3}


class Controller(Reporter):
    def __init__(self, experiments, ranks=1, mode="minimum", output=None):
        super().__init__(mode)
        self.output = Path(output) if output is not None else None
        self.ranks = ranks
        self._data = {key: self._data[key] for key in ("status", "started_at")}
        self._data["experiments"] = {
            name: {"status": "PENDING", "ranks": {}} for name in experiments}
        self.active = None
        self.process = None
        if mode != "zero":
            self.output.mkdir(parents=True, exist_ok=False)
            self.attach(self.output)

    def receive(self, name, rank, message):
        with self._lock:
            experiment = self._data["experiments"][name]
            reporter = experiment["ranks"][str(rank)]
            reporter.ingest(message)
            if message["type"] == "event":
                self.event("worker", experiment=name, rank=rank, entry=message["entry"])
            elif message["type"] == "update":
                self.event("worker_update", experiment=name, rank=rank, fields=message["fields"])

    def snapshot(self):
        with self._lock:
            # Rank reporters contain locks, so materialize them before copying.
            return {**deepcopy({key: value for key, value in self._data.items() if key != "experiments"}),
                    "experiments": {
                        name: {**deepcopy({key: value for key, value in entry.items() if key != "ranks"}),
                               "ranks": {rank: reporter.snapshot()
                                         for rank, reporter in entry["ranks"].items()}}
                        for name, entry in self._data["experiments"].items()}}

    def metrics(self):
        data = self.snapshot()
        # Reuse the worker metric schema; distinguish retained jobs by fixed labels.
        lines = []
        seen = set()
        for name, entry in data["experiments"].items():
            for rank, result in entry["ranks"].items():
                reporter = Reporter("zero")
                reporter.update(**result)
                for line in reporter.metrics().splitlines():
                    if line.startswith("#"):
                        if line not in seen:
                            lines.append(line)
                            seen.add(line)
                    else:
                        metric, value = line.rsplit(" ", 1)
                        labels = f'experiment="{name}",rank="{rank}"'
                        metric = (metric[:-1] + "," + labels + "}" if "{" in metric
                                  else metric + "{" + labels + "}")
                        lines.append(metric + " " + value)
        lines += ["# HELP tiny_experiments_controller_status Suite status: running=0 pass=1 fail=2 error=3.",
                  "# TYPE tiny_experiments_controller_status gauge",
                  f'tiny_experiments_controller_status {STATUS_VALUE[data["status"]]}']
        lines += ["# HELP tiny_experiments_job_status Job status: pending=-1 running=0 pass=1 fail=2 error=3.",
                  "# TYPE tiny_experiments_job_status gauge"]
        for name, entry in data["experiments"].items():
            value = STATUS_VALUE[entry["status"]]
            lines.append(f'tiny_experiments_job_status{{experiment="{name}"}} {value}')
        return "\n".join(lines) + "\n"

    @contextmanager
    def collect(self):
        controller = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                name = rank = None
                try:
                    hello = json.loads(self.rfile.readline())
                    name, rank = hello["experiment"], hello["rank"]
                    if hello["type"] != "hello" or name != controller.active or not 0 <= rank < controller.ranks:
                        raise ValueError("unexpected worker identity")
                    for line in self.rfile:
                        controller.receive(name, rank, json.loads(line))
                except (ValueError, KeyError, TypeError) as exc:
                    controller.event("collection_error", experiment=name, rank=rank, error=str(exc))
                    if name == controller.active and str(rank) in controller._data["experiments"][name]["ranks"]:
                        controller._data["experiments"][name]["ranks"][str(rank)].finish("ERROR", str(exc))

        class Server(socketserver.ThreadingUnixStreamServer):
            # server_close joins readers before final results are inspected.
            daemon_threads = False

        with tempfile.TemporaryDirectory(prefix="tiny-exp-") as directory:
            path = str(Path(directory) / "events.sock")
            with Server(path, Handler) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    yield path
                finally:
                    server.shutdown()
                    thread.join()

    def stop(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            # The launcher may have exited before its workers. Remove survivors too.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def run_suite(self, command_builder, timeout=0, job_options=None):
        """command_builder(name) returns argv; no shell parsing is used."""
        try:
            for name, entry in self._data["experiments"].items():
                with self._lock:
                    self.active = name
                    entry.update(status="RUNNING", started_at=timestamp(), ranks={
                        str(rank): Reporter("zero") for rank in range(self.ranks)})
                options = (job_options or {}).get(name, {})
                settings = {**options.get("config", options.get("settings", {})),
                            **{key: options[key] for key in ("seed", "repeats", "threads", "device") if key in options},
                            "ranks": self.ranks}
                self.experiment_log.start(name, settings)
                command = command_builder(name)
                with self._lock:
                    entry["command"] = command
                self.event("job_started", experiment=name, command=command)
                tail = ""
                error = None
                with self.collect() as channel:
                    env = {**os.environ, **options.get("environment", {}), "EXPERIMENT_CONTROLLER_SOCKET": channel,
                           "EXPERIMENT_LOGGING": self.mode}
                    def drain():
                        nonlocal tail
                        for chunk in iter(lambda: self.process.stdout.read(4096), ""):
                            tail = (tail + chunk)[-8192:]
                            self.event("launcher_output", debug=True, experiment=name, text=chunk)

                    reader = None
                    try:
                        self.process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                                        stderr=subprocess.STDOUT, text=True,
                                                        errors="replace", start_new_session=True)
                        reader = threading.Thread(target=drain, daemon=True)
                        reader.start()
                        code = self.process.wait(timeout=options.get("job_timeout", timeout) or None)
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        self.stop()
                        code = self.process.returncode if self.process is not None else -1
                    finally:
                        self.stop()
                        if reader is not None:
                            reader.join()
                        if self.process is not None and self.process.stdout is not None:
                            self.process.stdout.close()
                        self.process = None
                # All IPC connections have drained before deciding the job outcome.
                for rank, reporter in entry["ranks"].items():
                    if reporter.snapshot()["status"] == "RUNNING":
                        reporter.finish("ERROR", error or "worker exited without a final result")
                statuses = [reporter.snapshot()["status"] for reporter in entry["ranks"].values()]
                status = ("ERROR" if error or "ERROR" in statuses else
                          "FAIL" if "FAIL" in statuses else "ERROR" if code else "PASS")
                reason = error
                if status != "PASS" and reason is None:
                    for rank, reporter in entry["ranks"].items():
                        result = reporter.snapshot()
                        if result["status"] in ("FAIL", "ERROR"):
                            failed_test = next((test for test in result["tests"]
                                                if test["status"] in ("FAIL", "ERROR")), None)
                            detail = result.get("error") or (compact(failed_test) if failed_test else "no final result")
                            reason = f"rank {rank}: {detail}"
                            break
                with self._lock:
                    entry.update(status=status, exit_code=code, finished_at=timestamp())
                    if status != "PASS":
                        entry["error"] = reason or f"launcher exited with code {code}; inspect rank results"
                        entry["launcher_tail"] = tail
                self.event("job_finished", experiment=name, status=status, exit_code=code,
                           error=entry.get("error"), launcher_tail=tail if status != "PASS" else None)
                self.experiment_log.finish(name, status, entry.get("error"))
                if self.mode == "verbose":
                    for rank, reporter in entry["ranks"].items():
                        directory = self.output / name / f"rank-{int(rank):04d}"
                        if not directory.is_dir() and self.ranks == 1:
                            directory = self.output / name
                        if directory.is_dir():
                            snapshot = Reporter("minimum")
                            snapshot.update(**reporter.snapshot())
                            snapshot.write(directory)
                self.write(self.output)
            statuses = [entry["status"] for entry in self._data["experiments"].values()]
            self.finish("ERROR" if "ERROR" in statuses else "FAIL" if "FAIL" in statuses else "PASS")
        except BaseException as exc:
            self.stop()
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                if self.active is not None:
                    entry = self._data["experiments"][self.active]
                    entry.update(status="ERROR", error=error, finished_at=timestamp())
                    for reporter in entry["ranks"].values():
                        if reporter.snapshot()["status"] == "RUNNING":
                            reporter.finish("ERROR", error)
                    self.experiment_log.finish(self.active, "ERROR", error)
            self.finish("ERROR", error)
            raise
        finally:
            self.active = None
            self.experiment_log.summary()
            self.write(self.output)
        return int(self.snapshot()["status"] != "PASS")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*")
    parser.add_argument("--config", type=Path, help="YAML suite configuration")
    parser.add_argument("--check-config", action="store_true", help="validate settings without running workloads")
    parser.add_argument("--describe", metavar="EXPERIMENT", help="show configurable fields and defaults")
    parser.add_argument("--logging", choices=("zero", "minimum", "verbose"), default="minimum")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=9100)
    parser.add_argument("--api-linger", type=float, default=30)
    parser.add_argument("--job-timeout", type=float, default=0, help="per experiment seconds; zero disables")
    preliminary, _ = parser.parse_known_args(argv)
    config = {"controller": {}, "experiments": {}, "environment": {}}
    try:
        if preliminary.config:
            config = load_config(preliminary.config)
        parser.set_defaults(**config["controller"])
        args = parser.parse_args(argv)
        if args.describe:
            print(json.dumps(describe(args.describe), indent=2))
            return 0
        if not args.experiments:
            args.experiments = [name for name, entry in config["experiments"].items() if entry["enabled"]]
        if not args.experiments:
            parser.error("no enabled experiments; supply --config or experiment names")
        validate_run({key: getattr(args, key) for key in config["controller"] if key != "output"}, "controller")
    except (ValueError, ImportError, OSError) as exc:
        parser.error(str(exc))
    if len(set(args.experiments)) != len(args.experiments) or any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in args.experiments):
        parser.error("experiment names must be unique simple module names")
    if not args.check_config and args.logging != "zero" and args.output is None:
        parser.error("--output is required for minimum/verbose logging")
    if min(args.nproc_per_node, args.threads, args.repeats) < 1 or not 0 <= args.seed < 2**63:
        parser.error("ranks, threads, repeats must be positive and seed in [0, 2**63)")
    if not 0 <= args.api_port <= 65535 or any(not math.isfinite(value) or value < 0
                                            for value in (args.api_linger, args.job_timeout)):
        parser.error("invalid port, linger or timeout")
    if args.device == "cpu" and args.nproc_per_node != 1:
        parser.error("CPU execution requires --nproc-per-node 1")
    jobs = {}
    try:
        for name in args.experiments:
            entry = config["experiments"].get(name, {})
            jobs[name] = {"settings": entry.get("settings", {}),
                          "device": args.device,
                          "environment": {**config["environment"], **entry.get("environment", {})},
                          **{key: getattr(args, key) for key in ("seed", "threads", "repeats", "job_timeout")},
                          **entry.get("run", {})}
            env = {**os.environ, **jobs[name]["environment"]}
            inspector = env.get("NCCL_INSPECTOR_ENABLE", "0")
            jobs[name]["environment"] = {
                "NCCL_INSPECTOR_DUMP_THREAD_ENABLE": inspector,
                "NCCL_INSPECTOR_PROM_DUMP": inspector,
                "NCCL_INSPECTOR_DUMP_VERBOSE": "1" if args.logging == "verbose" else "0",
                **jobs[name]["environment"]}
            # Resolve defaults for the readable log; YAML/check-config always validate before launch.
            if args.config or args.check_config or args.logging == "minimum":
                jobs[name]["config"] = json.loads(compact(asdict(load_experiment(name, jobs[name]["settings"]))))
    except (ValueError, ImportError, AttributeError, TypeError) as exc:
        parser.error(f"{name}: {exc}")
    if args.check_config:
        print("Configuration valid; no experiments executed.")
        return 0
    controller = Controller(args.experiments, args.nproc_per_node, args.logging, args.output)

    def command(name):
        worker = [str(Path(__file__).with_name("worker.py")), name, "--device", args.device,
                  "--threads", str(jobs[name]["threads"]), "--seed", str(jobs[name]["seed"]),
                  "--repeats", str(jobs[name]["repeats"]), "--settings-json", compact(jobs[name]["settings"])]
        if args.output is not None:
            worker += ["--output", str(args.output / name)]
        launch = ([sys.executable] if args.device == "cpu" else
                  [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc-per-node={args.nproc_per_node}", "--max-restarts=0"])
        job_env = {**os.environ, **jobs[name]["environment"]}
        if args.logging == "verbose" and job_env.get("PROFILER_ACTIVE") == "1":
            profile = Path(job_env.get("PROFILE", "/profiles")) / name
            profile.mkdir(parents=True, exist_ok=False)
            launch = ["nsys", "profile", "--trace=cuda,nvtx,nccl", "--nccl-trace=api,rt,gpu",
                      "--sample=none", "--cpuctxsw=none", "--trace-fork-before-exec=false",
                      f"--output={profile}/%h-%p"] + launch
        return launch + worker

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        with controller.serve(args.api_host, args.api_port):
            try:
                code = controller.run_suite(command, args.job_timeout, jobs)
            except KeyboardInterrupt:
                code = 130
            except Exception:
                # run_suite records controller/setup failures before returning here.
                code = 1
            if args.api_linger:
                time.sleep(args.api_linger)
            return code
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())

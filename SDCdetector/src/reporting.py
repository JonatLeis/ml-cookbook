"""Reporting and logging for experiments, with optional HTTP metrics endpoint"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time


FAILURE_EVENTS = {"SDC_CANDIDATE", "SDC_CANDIDATE_SUMMARY", "MISMATCH",
                  "INVALID_TRAINING_STATE", "SCHEMA_MISMATCH", "FAIL"}

_current = ContextVar("reporter", default=None)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def compact(value):
    return json.dumps(value, separators=(",", ":"), default=str)


class ExperimentLog:
    """A readable experiment lifecycle log; inactive unless a path is attached."""

    def __init__(self):
        self.path = None
        self.experiments = {}

    def _append(self, text):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def start(self, name, settings):
        if self.path is None or name in self.experiments:
            return
        setting = ", ".join(f"{key}={compact(value)}" for key, value in settings.items()) or "defaults"
        self.experiments[name] = {"setting": setting, "status": "RUNNING"}
        self._append(f"[{timestamp()}] Running Experiment {name} with setting: {setting}...")

    def finish(self, name, status, error=None):
        if self.path is None:
            return
        entry = self.experiments[name]
        if entry["status"] != "RUNNING":
            return
        entry["status"] = "PASSED" if status == "PASS" else "FAILED"
        self._append(f"[{timestamp()}] Experiment {name} {entry['status']}!")
        if error:
            # Keep multiline exception messages from disrupting the table/log layout.
            self._append("  Reason: " + " ".join(error.split()))

    def summary(self):
        if self.path is None:
            return
        # Stable sorting preserves execution order within each outcome group.
        rows = []
        for name, entry in sorted(self.experiments.items(), key=lambda item: item[1]["status"] != "PASSED"):
            setting = entry["setting"]
            if len(setting) > 120:
                setting = setting[:117] + "..."
            rows.append((f"{name} ({setting})", entry["status"]))
        width = max([len("Experiment (setting)")] + [len(label) for label, _ in rows])
        lines = [f"\n[{timestamp()}] Experiment summary (PASSED first; execution order within each group):",
                 f"{'Experiment (setting)':<{width}} | Result",
                 f"{'-' * width}-+--------"]
        lines.extend(f"{label:<{width}} | {status}" for label, status in rows)
        self._append("\n".join(lines))


class Reporter:
    """One run per reporter. snapshot() and metrics() are safe during execution."""

    def __init__(self, mode="minimum", sink=None):
        if mode not in ("zero", "minimum", "verbose"):
            raise ValueError("mode must be zero, minimum or verbose")
        self.mode = mode
        self.sink = sink
        self.managed = sink is not None
        self.repetition = 0
        self._lock = threading.RLock()
        self._data = {"status": "RUNNING", "started_at": timestamp(), "metadata": {},
                      "completed_repeats": 0, "tests": [], "failures": []}
        self._log = None
        self.experiment_log = ExperimentLog()

    @contextmanager
    def activate(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def attach(self, output):
        if self.mode == "minimum":
            self.experiment_log.path = Path(output) / "run.log"
        elif self.mode == "verbose":
            self._log = Path(output) / "events.jsonl"

    def update(self, **fields):
        with self._lock:
            self._data.update(fields)
            if self.sink is not None:
                self.sink({"type": "update", "fields": fields})

    def snapshot(self):
        with self._lock:
            return deepcopy(self._data)

    def event(self, event, *, debug=False, **fields):
        if debug and self.mode != "verbose":
            return
        entry = {"timestamp": timestamp(), "event": event,
                 "repetition": self.repetition, **fields}
        with self._lock:
            if self.sink is not None:
                self.sink({"type": "event", "entry": entry})
            if self._log is not None:
                with self._log.open("a") as stream:
                    stream.write(compact(entry) + "\n")

    def test(self, name, status, **details):
        entry = {"name": name, "status": status, "timestamp": timestamp(),
                 "repetition": self.repetition}
        if status in ("FAIL", "ERROR"):
            entry["details"] = details
        with self._lock:
            self._data["tests"].append(entry)
            self.event("test", **entry)

    def diagnostic(self, event, **fields):
        failure = event in FAILURE_EVENTS
        with self._lock:
            if failure:
                self._data["failures"].append({"event": event, "timestamp": timestamp(),
                                               "repetition": self.repetition, **fields})
            self.event(event, debug=not failure, **fields)

    def ingest(self, message):
        """Apply a worker's structured update without changing its timestamps."""
        with self._lock:
            if message["type"] == "update":
                self._data.update(message["fields"])
            elif message["type"] == "event":
                entry = message["entry"]
                if entry["event"] == "test":
                    self._data["tests"].append({key: value for key, value in entry.items()
                                                if key != "event"})
                elif entry["event"] in FAILURE_EVENTS:
                    self._data["failures"].append(entry)
            else:
                raise ValueError("unknown worker message type")

    def finish(self, status, error=None):
        self.update(status=status, finished_at=timestamp())
        if error is not None:
            self.update(error=error)
        self.event("run_finished", status=status, error=error)

    def write(self, output):
        if self.mode != "zero":
            target = Path(output) / "report.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(compact(self.snapshot()) + "\n")
            temporary.replace(target)

    def metrics(self):
        """Run metrics and latest test outcomes; reproduction details stay in JSON."""
        data = self.snapshot()
        lines = []

        def gauge(name, help_text, value):
            full = "tiny_experiments_" + name
            lines.extend([f"# HELP {full} {help_text}", f"# TYPE {full} gauge",
                          f"{full} {value}"])

        # Status is one numeric gauge, avoiding dynamic exception/config labels.
        gauge("run_status", "Run status: running=0 pass=1 fail=2 error=3.",
              {"RUNNING": 0, "PASS": 1, "FAIL": 2, "ERROR": 3}[data["status"]])
        gauge("completed_repeats", "Successfully completed repetitions.", data["completed_repeats"])
        gauge("tests", "Recorded test outcomes in this run.", len(data["tests"]))
        gauge("failed_tests", "Failed or errored test outcomes in this run.",
              sum(test["status"] in ("FAIL", "ERROR") for test in data["tests"]))
        for field in ("started_at", "finished_at"):
            if field in data:
                gauge(field + "_seconds", "Unix time of run " + field + ".",
                      datetime.fromisoformat(data[field]).timestamp())
        # Keep only the latest outcome per named test; repetitions are not labels.
        latest = {test["name"]: test for test in data["tests"]}
        for name, help_text in (("test_status", "Latest test status: pass=1 fail=2 error=3."),
                                ("test_timestamp_seconds", "Unix time of latest test outcome.")):
            full = "tiny_experiments_" + name
            lines.extend([f"# HELP {full} {help_text}", f"# TYPE {full} gauge"])
            for case, test in latest.items():
                escaped = case.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                value = ({"PASS": 1, "FAIL": 2, "ERROR": 3}[test["status"]]
                         if name == "test_status" else datetime.fromisoformat(test["timestamp"]).timestamp())
                lines.append(f'{full}{{test="{escaped}"}} {value}')
        return "\n".join(lines) + "\n"

    @contextmanager
    def serve(self, host="127.0.0.1", port=0):
        """Serve until the context exits; keep it open to scrape completed runs."""
        reporter = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    body = reporter.metrics().encode()
                    content_type = "text/plain; version=0.0.4; charset=utf-8"
                elif self.path == "/results":
                    body = compact(reporter.snapshot()).encode()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer((host, port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


def debug(message):
    reporter = _current.get()
    if reporter is not None:
        reporter.event("trace", debug=True, message=message)


def diagnostic(event, **fields):
    reporter = _current.get()
    if reporter is not None:
        reporter.diagnostic(event, **fields)


def record_test(name, status, **details):
    reporter = _current.get()
    if reporter is not None:
        reporter.test(name, status, **details)


@contextmanager
def progress(name, kind="CHECK"):
    started = time.perf_counter()
    debug(f"{kind}_START {name}")
    try:
        yield
    except BaseException:
        debug(f"{kind}_ERROR {name}")
        raise
    finally:
        debug(f"{kind}_END {name} elapsed_s={time.perf_counter() - started:.6f}")

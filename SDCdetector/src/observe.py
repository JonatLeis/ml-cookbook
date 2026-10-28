"""Independent polling collector and live dashboard for a controller endpoint. - Full GPT!"""
import argparse
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from prometheus_client.parser import text_string_to_metric_families

from reporting import compact, timestamp


def parse_metrics(text):
    samples = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            key = sample.name + (" " + compact(dict(sorted(sample.labels.items()))) if sample.labels else "")
            samples[key] = {"name": sample.name, "labels": sample.labels,
                            "value": sample.value if math.isfinite(sample.value) else None}
    return samples


class Collector:
    def __init__(self, target, interval=2, history=300, timeout=3, output=None):
        parts = urlsplit(target)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.query or parts.fragment:
            raise ValueError("target must be an HTTP(S) controller base URL")
        if not math.isfinite(interval) or interval <= 0 or not math.isfinite(timeout) or timeout <= 0 or history < 1:
            raise ValueError("interval, timeout and history must be positive")
        self.target = target.rstrip("/")
        self.interval, self.timeout = interval, timeout
        self.history = deque(maxlen=history)
        self.results = None
        self.samples = {}
        self.last_success = None
        self.error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.output = Path(output) if output is not None else None
        self._last_report = None
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=False)

    def fetch(self, path):
        with urlopen(self.target + path, timeout=self.timeout) as response:
            return response.read().decode("utf-8")

    def collect_once(self):
        now = time.time()
        metrics, results = None, None
        errors = []
        try:
            metrics = parse_metrics(self.fetch("/metrics"))
        except (OSError, URLError, ValueError) as exc:
            errors.append(f"metrics: {exc}")
        try:
            results = json.loads(self.fetch("/results"))
            if not isinstance(results, dict) or "experiments" not in results:
                raise ValueError("expected a controller results report")
        except (OSError, URLError, ValueError) as exc:
            errors.append(f"results: {exc}")
            results = None
        sample = {"time": now, "metrics": metrics, "error": "; ".join(errors) or None}
        with self._lock:
            self.history.append(sample)
            self.error = sample["error"]
            if metrics is not None:
                self.samples = metrics
                self.last_success = timestamp()
            if results is not None:
                self.results = results
            if self.output is not None:
                with (self.output / "samples.jsonl").open("a") as stream:
                    stream.write(compact(sample) + "\n")
                report = compact(results) if results is not None else None
                if report is not None and report != self._last_report:
                    with (self.output / "results.jsonl").open("a") as stream:
                        stream.write(compact({"collected_at": timestamp(), "results": results}) + "\n")
                    self._last_report = report
        return sample

    def state(self, series=None):
        with self._lock:
            latest = self.samples
            if series is None:
                series = next((key for key in latest if "failed_tests" in key), next(iter(latest), None))
            return deepcopy({"target": self.target, "interval": self.interval,
                             "last_success": self.last_success, "error": self.error,
                             "results": self.results, "samples": latest, "series": series,
                             "history": [{"time": entry["time"],
                                          "value": (entry["metrics"] or {}).get(series, {}).get("value")}
                                         for entry in self.history]})

    @contextmanager
    def polling(self):
        def poll():
            while not self._stop.is_set():
                try:
                    self.collect_once()
                except OSError as exc:
                    # Persistence errors must be visible instead of killing the poller silently.
                    with self._lock:
                        self.error = f"collector storage: {exc}"
                self._stop.wait(self.interval)
        self._stop.clear()
        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        try:
            yield
        finally:
            self._stop.set()
            thread.join()

    @contextmanager
    def serve(self, host="127.0.0.1", port=9200):
        collector = self
        page = Path(__file__).with_name("observe.html").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlsplit(self.path)
                if url.path == "/":
                    body, content_type = page, "text/html; charset=utf-8"
                elif url.path == "/api/state":
                    series = parse_qs(url.query).get("series", [None])[0]
                    body, content_type = compact(collector.state(series)).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer((host, port), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield server.server_address
            finally:
                server.shutdown()
                thread.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="http://127.0.0.1:9100")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=3)
    parser.add_argument("--history", type=int, default=300, help="maximum scrapes retained in memory")
    parser.add_argument("--output", type=Path, help="new directory for optional durable JSONL collection")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")
    try:
        collector = Collector(args.target, args.interval, args.history, args.timeout, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        with collector.serve(args.host, args.port) as (host, port), collector.polling():
            print(f"Observability dashboard: http://{host}:{port} — target: {collector.target}", flush=True)
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

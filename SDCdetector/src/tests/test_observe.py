"""Independent collector/parser/dashboard checks with a local controller fixture."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from controller import Controller
from observe import Collector, parse_metrics
from reporting import Reporter


class ObserverTests(unittest.TestCase):
    def source(self):
        controller = Controller(["example"], mode="zero")
        reporter = Reporter("zero")
        reporter.test('tensor"key', "FAIL", element=3)
        reporter.finish("FAIL")
        controller._data["experiments"]["example"].update(status="FAIL", ranks={"0": reporter})
        controller.finish("FAIL")
        return controller

    def test_parses_labels_and_nonfinite_samples(self):
        metrics = parse_metrics('metric{case="a\\\"b"} 3\nother NaN\n')
        self.assertEqual(next(iter(metrics.values()))["labels"]["case"], 'a"b')
        self.assertIsNone(metrics["other"]["value"])

    def test_collection_persistence_and_dashboard_api(self):
        with tempfile.TemporaryDirectory() as directory, self.source().serve() as (host, port):
            output = Path(directory) / "collected"
            collector = Collector(f"http://{host}:{port}", history=2, output=output)
            collector.collect_once()
            collector.collect_once()
            collector.collect_once()
            state = collector.state()
            self.assertIsNone(state["error"])
            self.assertEqual(state["results"]["status"], "FAIL")
            self.assertEqual(len(state["history"]), 2)
            series = state["series"]
            self.assertEqual(state["samples"][series]["value"], 1)
            self.assertEqual(len((output / "samples.jsonl").read_text().splitlines()), 3)
            self.assertEqual(len((output / "results.jsonl").read_text().splitlines()), 1)
            with collector.serve(port=0) as (web_host, web_port):
                with urlopen(f"http://{web_host}:{web_port}/api/state") as response:
                    self.assertEqual(json.load(response)["results"]["status"], "FAIL")
                with urlopen(f"http://{web_host}:{web_port}/") as response:
                    self.assertIn("Metric history", response.read().decode())

    def test_outage_retains_last_data_and_recovers(self):
        controller = self.source()
        collector = Collector("http://localhost:9100", history=1)
        def fetch(path):
            return controller.metrics() if path == "/metrics" else json.dumps(controller.snapshot())
        with patch.object(collector, "fetch", side_effect=fetch):
            collector.collect_once()
        original = collector.state()["samples"]
        with patch.object(collector, "fetch", side_effect=OSError("disconnected")):
            collector.collect_once()
            collector.collect_once()
        state = collector.state()
        self.assertIn("disconnected", state["error"])
        self.assertEqual(state["samples"], original)
        self.assertEqual(state["results"]["status"], "FAIL")
        self.assertIsNone(state["history"][0]["value"])
        with patch.object(collector, "fetch", side_effect=fetch):
            collector.collect_once()
        self.assertIsNone(collector.state()["error"])

    def test_validation(self):
        for kwargs in ({"target": "file:///tmp/test"}, {"target": "http://localhost", "history": 0},
                       {"target": "http://localhost", "interval": -1}):
            with self.assertRaises(ValueError):
                Collector(**kwargs)

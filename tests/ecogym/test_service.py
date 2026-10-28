"""HTTP service on the AGC task with a scratch ledger (development stages only)."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from ecogym.service import Service, make_server

from .helpers import TASK_ID, requires_agc

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@requires_agc
class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.service = Service(Path(cls.directory.name) / "ledger.sqlite3")
        cls.server = make_server(cls.service, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.directory.cleanup()

    def call(self, path, body=None, *, header=True):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method="POST" if body is not None else "GET")
        request.add_header("Content-Type", "application/json")
        if header and body is not None:
            request.add_header("X-Ecogym", "1")
        try:
            with NO_PROXY.open(request, timeout=60) as response:
                raw = response.read().decode()
                return response.status, json.loads(raw) if "json" in response.headers["Content-Type"] else raw
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode())

    def test_workbench_and_catalog(self):
        status, page = self.call("/")
        self.assertEqual(status, 200)
        self.assertIn("ecogym", page)
        status, catalog = self.call("/api/catalog")
        self.assertIn(TASK_ID, [task["id"] for task in catalog["tasks"]])
        self.assertIn("llm", catalog["proposers"])

    def test_run_lifecycle(self):
        status, started = self.call("/api/runs", {"task": TASK_ID, "preset": "recipe_seed", "proposer": "random",
                                                  "generations": 2, "label": "service test"})
        self.assertEqual(status, 201)
        run_id = started["run_id"]
        for _ in range(200):
            status, run = self.call(f"/api/runs/{run_id}")
            if not run["active"] and run["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(run["status"], "finished")
        self.assertEqual(len(run["generations"]), 2)
        status, cells = self.call(f"/api/runs/{run_id}/cells")
        self.assertEqual(len(cells["final"]["model_selection"]["cells"]), 9)
        status, page = self.call(f"/api/runs/{run_id}/report")
        self.assertIn("Generation 1", page)
        status, runs = self.call("/api/runs")
        self.assertIn(run_id, [item["run_id"] for item in runs])

    def test_posts_need_the_header_and_valid_bodies(self):
        status, _ = self.call("/api/runs", {"task": TASK_ID, "preset": "expert"}, header=False)
        self.assertEqual(status, 403)
        status, _ = self.call("/api/runs", {"task": TASK_ID, "preset": "nope"})
        self.assertEqual(status, 400)
        status, _ = self.call("/api/runs/run:missing")
        self.assertEqual(status, 404)
        status, _ = self.call("/api/runs/run:missing/seal", {"stage": "validation"})
        self.assertEqual(status, 404)

    def test_non_loopback_needs_a_token(self):
        with self.assertRaises(ValueError):
            make_server(self.service, "0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()

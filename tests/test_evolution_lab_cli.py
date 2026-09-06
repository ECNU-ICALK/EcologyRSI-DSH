import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ecologyrsi_dsh.evolution_lab.__main__ import main


class EvolutionLabCliTests(unittest.TestCase):
    def test_demo_creates_and_promotes_a_plugin_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = StringIO()
            with redirect_stdout(output):
                code = main(["--demo", "--db", str(Path(temp) / "lab.sqlite")])
            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["decision"]["status"], "certification_eligible")
            self.assertEqual(payload["incumbent_digest"], payload["candidate_digest"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest

from scripts.runtime_gc import plan_cleanup, run_cleanup


class RuntimeGcTests(unittest.TestCase):
    def test_plan_is_bounded_to_old_regular_files_and_apply_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".runtime"
            root.mkdir()
            old_log = root / "old.log"
            old_log.write_text("old", encoding="utf-8")
            os.utime(old_log, (time.time() - 10 * 86400,) * 2)
            fresh_log = root / "fresh.log"
            fresh_log.write_text("fresh", encoding="utf-8")
            active_db = root / "active.sqlite3"
            active_db.write_text("db", encoding="utf-8")

            planned = plan_cleanup(root, older_than_days=7, keep=0)
            self.assertEqual([item["path"] for item in planned], ["old.log"])
            self.assertTrue(old_log.exists())
            result = run_cleanup(root, older_than_days=7, keep=0, apply=True)
            self.assertEqual(result["deleted"], ["old.log"])
            self.assertFalse(old_log.exists())
            self.assertTrue(fresh_log.exists())
            self.assertTrue(active_db.exists())

    def test_symlink_is_rejected_even_when_old(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".runtime"
            root.mkdir()
            target = root / "target.log"
            target.write_text("target", encoding="utf-8")
            link = root / "old.log"
            link.symlink_to(target)
            with self.assertRaises(RuntimeError):
                plan_cleanup(root, older_than_days=0, keep=0)


if __name__ == "__main__":
    unittest.main()

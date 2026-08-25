from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ecologyrsi_dsh.application.cli import _bundled_dsh_plugin_archive
from ecologyrsi_dsh.version import __version__


class DshRuntimeInstallTests(unittest.TestCase):
    def test_exact_current_archive_wins_over_historical_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plugin_root = Path(directory)
            dist = plugin_root / "dist"
            dist.mkdir()
            (dist / "ecologyrsi-dsh-evolution-plugin-0.3.1.tgz").touch()
            expected = dist / f"ecologyrsi-dsh-evolution-plugin-{__version__}.tgz"
            expected.touch()

            self.assertEqual(_bundled_dsh_plugin_archive(plugin_root), expected)

    def test_missing_current_archive_is_not_replaced_by_old_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plugin_root = Path(directory)
            dist = plugin_root / "dist"
            dist.mkdir()
            (dist / "ecologyrsi-dsh-evolution-plugin-0.3.1.tgz").touch()

            with self.assertRaisesRegex(RuntimeError, __version__):
                _bundled_dsh_plugin_archive(plugin_root)


if __name__ == "__main__":
    unittest.main()

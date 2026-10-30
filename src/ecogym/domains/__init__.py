"""Domain packs: datasets, task cards, task suites and domain solvers.

A pack is a module exposing ``DATASETS``, ``TASKS`` and optionally ``SUITES``
(paths to TOML files) and registering its solvers on import. Built-in packs
live here; external packs are discovered through the ``ecogym.domains``
entry-point group. Suites are checked once every pack is loaded, so a suite
may combine tasks of several packs.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from types import ModuleType

from ..data.catalog import dataset_spec, register_dataset
from ..task.spec import check_task, register_task
from ..task.suite import check_suite, register_suite

BUILTIN = ("ecogym.domains.greenhouse",)
_LOADED: dict[str, ModuleType] = {}


def _register(module: ModuleType) -> list:
    for path in getattr(module, "DATASETS", ()):
        register_dataset(path)
    for path in getattr(module, "TASKS", ()):
        task = register_task(path)
        check_task(task, dataset_spec(task.dataset))
    return [register_suite(path) for path in getattr(module, "SUITES", ())]


def load_domains() -> list[str]:
    """Import every built-in and installed pack once; return their names."""

    names = list(BUILTIN) + [point.value for point in entry_points(group="ecogym.domains")]
    suites = []
    for name in names:
        if name not in _LOADED:
            module = importlib.import_module(name)
            suites.extend(_register(module))
            _LOADED[name] = module
    for suite in suites:
        check_suite(suite)
    return list(_LOADED)


__all__ = ["BUILTIN", "load_domains"]

"""Domain packs: datasets, task cards and domain solvers.

A pack is a module exposing ``DATASETS`` and ``TASKS`` (paths to TOML files)
and registering its solvers on import. Built-in packs live here; external
packs are discovered through the ``ecogym.domains`` entry-point group.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from types import ModuleType

from ..data.catalog import dataset_spec, register_dataset
from ..task.spec import check_task, register_task

BUILTIN = ("ecogym.domains.greenhouse",)
_LOADED: dict[str, ModuleType] = {}


def _register(module: ModuleType) -> None:
    for path in getattr(module, "DATASETS", ()):
        register_dataset(path)
    for path in getattr(module, "TASKS", ()):
        task = register_task(path)
        check_task(task, dataset_spec(task.dataset))


def load_domains() -> list[str]:
    """Import every built-in and installed pack once; return their names."""

    names = list(BUILTIN) + [point.value for point in entry_points(group="ecogym.domains")]
    for name in names:
        if name not in _LOADED:
            module = importlib.import_module(name)
            _register(module)
            _LOADED[name] = module
    return list(_LOADED)


__all__ = ["BUILTIN", "load_domains"]

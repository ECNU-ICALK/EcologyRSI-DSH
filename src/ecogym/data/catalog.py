"""Dataset registry, data-root resolution and a load cache.

Domain packs register their dataset specs here. Source files live under a
data root: ``$ECOGYM_DATA_ROOT`` if set, otherwise the first existing of
``~/.ecogym/data`` and the pre-ecogym location ``~/.ecologyrsi-dsh/data``.
Parsed frames are cached as ``.npz`` under ``$ECOGYM_CACHE_DIR`` (default
``~/.cache/ecogym``), keyed by the spec digest, the loader version and the
size and modification time of every source file.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..kernel.canonical import digest
from .frame import Frame, _frame_digest, _readonly
from .loaders import LOADERS, source_files
from .spec import DatasetSpec, load_dataset_spec

LOADER_VERSION = "ecogym.loaders/1"
_SPECS: dict[str, DatasetSpec] = {}


def register_dataset(path: str | Path) -> DatasetSpec:
    spec = load_dataset_spec(path)
    existing = _SPECS.get(spec.id)
    if existing is not None and existing.digest != spec.digest:
        raise ValueError(f"dataset {spec.id} is registered twice with different specs")
    _SPECS[spec.id] = spec
    return spec


def dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return _SPECS[dataset_id]
    except KeyError:
        raise KeyError(f"unknown dataset {dataset_id}; known: {', '.join(sorted(_SPECS)) or 'none'}") from None


def dataset_specs() -> list[DatasetSpec]:
    return [_SPECS[key] for key in sorted(_SPECS)]


def data_root() -> Path:
    configured = os.environ.get("ECOGYM_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    candidates = [Path.home() / ".ecogym" / "data", Path.home() / ".ecologyrsi-dsh" / "data"]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def dataset_dir(spec: DatasetSpec, root: Path | None = None) -> Path:
    return (root or data_root()) / spec.path


def _cache_dir() -> Path:
    return Path(os.environ.get("ECOGYM_CACHE_DIR") or Path.home() / ".cache" / "ecogym").expanduser()


def _cache_key(spec: DatasetSpec, directory: Path) -> str:
    files = []
    for path in source_files(spec, directory):
        stat = path.stat()
        files.append([str(path.relative_to(directory)), stat.st_size, stat.st_mtime_ns])
    return digest({"spec": spec.digest, "loader": LOADER_VERSION, "files": files})


def _save(frame: Frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(frame.columns)
    values = np.stack([frame.columns[name] for name in names], axis=1) if names else np.zeros((len(frame), 0))
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        time=frame.time,
        episode=frame.episode,
        present=frame.present,
        episode_ids=np.array(frame.episode_ids, dtype=str),
        names=np.array(names, dtype=str),
        values=values,
        digest=np.array(frame.digest),
    )
    temporary.replace(path)


def _load(path: Path, spec: DatasetSpec) -> Frame | None:
    try:
        with np.load(path, allow_pickle=False) as stored:
            names = [str(name) for name in stored["names"]]
            values = stored["values"]
            columns = {name: _readonly(np.ascontiguousarray(values[:, index])) for index, name in enumerate(names)}
            time = _readonly(stored["time"].astype(np.int64))
            episode = _readonly(stored["episode"].astype(np.int32))
            present = _readonly(stored["present"].astype(bool))
            episode_ids = tuple(str(name) for name in stored["episode_ids"])
            expected = str(stored["digest"])
    except (OSError, KeyError, ValueError):
        return None
    specs = {name: spec.column(name) for name in names}
    actual = _frame_digest(spec.id, spec.step, episode_ids, time, episode, present, columns, specs)
    if actual != expected:
        return None
    return Frame(spec.id, spec.step, time, episode, episode_ids, present, columns, specs, actual)


def load_frame(dataset: str | DatasetSpec, *, root: Path | None = None, cache: bool = True) -> Frame:
    spec = dataset if isinstance(dataset, DatasetSpec) else dataset_spec(dataset)
    directory = dataset_dir(spec, root)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"data for {spec.id} not found at {directory}; set ECOGYM_DATA_ROOT to the directory holding {spec.path}"
        )
    try:
        loader = LOADERS[spec.loader]
    except KeyError:
        raise ValueError(f"{spec.id} names unknown loader {spec.loader}") from None
    cached = _cache_dir() / f"{spec.id}-{_cache_key(spec, directory)[:20]}.npz" if cache else None
    if cached is not None and cached.is_file():
        frame = _load(cached, spec)
        if frame is not None:
            return frame
    frame = loader(spec, directory)
    if cached is not None:
        _save(frame, cached)
    return frame


__all__ = [
    "LOADER_VERSION",
    "data_root",
    "dataset_dir",
    "dataset_spec",
    "dataset_specs",
    "load_frame",
    "register_dataset",
]

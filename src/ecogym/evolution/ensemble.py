"""Averaged forecasts of a run's episodes: what the spread between independent attempts is worth.

Episodes of one harness version on one task are independent attempts: each agent writes its own program, and
the judge stores its hidden-period forecasts. Averaging them origin by origin shows how much of the episodes'
errors are independent of each other. Only members that forecast every scored origin take part, so the gain
over the members' mean cannot come from filling one member's gaps with the others' forecasts. The scores are
read from the record and never reach the adoption gate, which compares single episodes because a harness is
judged by what one agent does with it. They show which forecaster a run could ship, and tell the evolver
whether the programs of different episodes complement each other.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..episodes.judge import average, load_predictions
from ..evaluation.harness import Bench, Evaluation
from .fold import SCORED_STATUSES

EVERY = "all"  # the key of every version's episodes together


def ensemble_scores(view: dict[str, Any], bench: Callable[[str], Bench], *,
                    failures: list[str] | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """``{harness digest or "all": {task: {"n", "score", "mean", "left_out"}}}`` for a folded evolution run.

    ``n`` members forecast every origin; ``score`` is their averaged forecasts' objective and ``mean`` the mean
    of their own; ``left_out`` counts the group's episodes with missing forecasts. Only groups of at least two
    complete members, and "all" only where a task has episodes of several versions. Members whose stored
    forecasts cannot be read (moved, changed, judged on other data) are skipped and named in ``failures``.
    """

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for round_ in view["rounds"]:
        for entry in round_["episodes"]:
            if entry.get("status") in SCORED_STATUSES and entry.get("predictions"):
                for key in (entry["harness"], EVERY):
                    groups.setdefault((key, entry["task"]), []).append(entry)
    loaded: dict[str, Evaluation | None] = {}

    def member(entry: dict[str, Any]) -> Evaluation | None:
        path = entry["predictions"]
        if path not in loaded:
            try:
                loaded[path] = load_predictions(Path(path), bench(entry["task"]),
                                                expect=entry.get("predictions_digest"))
            except (OSError, KeyError, ValueError) as error:
                loaded[path] = None
                if failures is not None:
                    failures.append(f"{path}: {type(error).__name__}: {error}")
        return loaded[path]

    found: dict[str, dict[str, dict[str, Any]]] = {}
    for (key, task), entries in groups.items():
        if key == EVERY and len({entry["harness"] for entry in entries}) < 2:
            continue
        members = [item for item in map(member, entries) if item is not None]
        complete = [item for item in members if np.isfinite(item.predicted).all()]
        if len(complete) < 2:
            continue
        found.setdefault(key, {})[task] = {"n": len(complete), "score": average(complete).score(),
                                           "mean": float(np.mean([item.score() for item in complete])),
                                           "left_out": len(members) - len(complete)}
    return found


__all__ = ["EVERY", "ensemble_scores"]
